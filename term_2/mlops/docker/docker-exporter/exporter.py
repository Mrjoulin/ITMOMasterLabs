"""
Docker container metrics exporter for Prometheus.

Reads live container stats from the Docker API and exposes them using
the same metric names that cAdvisor uses, so Grafana dashboards designed
for cAdvisor work without modification.

Works on cgroup v1 and v2 systems where cAdvisor cannot see container cgroups.
"""

import os
import time
import logging

import docker
from prometheus_client.core import (
    GaugeMetricFamily,
    CounterMetricFamily,
    REGISTRY,
)
from prometheus_client import start_http_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("docker-exporter")


def _short_id(container_id: str) -> str:
    """Return the 12-char short container id."""
    return container_id[:12]


def _container_labels(container) -> dict[str, str]:
    """Build a consistent label dict for a container."""
    name = container.name
    short = _short_id(container.id)
    image_tag = container.image.tags[0] if container.image.tags else container.image.short_id
    return {"name": name, "container": name, "id": short, "image": image_tag}


class DockerCollector:
    """Collects Docker container stats and yields Prometheus MetricFamily objects.

    Uses the collector interface from prometheus_client.core so that label
    values are always correctly registered.
    """

    def __init__(self, interval: float = 10.0) -> None:
        self.interval = interval
        self.client = docker.from_env()

    # -- internal accumulators for counter deltas ---------------------------
    # Counters must always increase, so we track previous raw values and
    # emit the accumulated delta.

    def collect(self):  # noqa: C901 – intentionally long
        """Yield prometheus_client.core.MetricFamily objects."""
        try:
            containers = self.client.containers.list()
        except docker.errors.APIError as exc:
            log.error("Failed to list containers: %s", exc)
            return

        # ---- metric accumulators keyed by label-tuple ----------------------
        cpu_usage: dict[tuple, float] = {}        # label vals -> seconds
        mem_working: dict[tuple, float] = {}
        mem_usage: dict[tuple, float] = {}
        net_rx: dict[tuple, float] = {}
        net_tx: dict[tuple, float] = {}
        fs_read: dict[tuple, float] = {}
        fs_write: dict[tuple, float] = {}

        # We also need the ordered label names for MetricFamily
        base_lbl = ["name", "container", "id", "image"]

        for container in containers:
            try:
                stats = container.stats(stream=False)
            except docker.errors.APIError:
                continue

            cl = _container_labels(container)
            lbl_vals = tuple(cl[k] for k in base_lbl)

            # -- CPU ---------------------------------------------------------
            cpu_stats = stats.get("cpu_stats") or {}
            percpu = (cpu_stats.get("cpu_usage") or {}).get("percpu_usage") or []
            for i, usage in enumerate(percpu):
                key = lbl_vals + (str(i),)
                cpu_usage[key] = usage / 1e9  # nanoseconds → seconds

            # -- Memory ------------------------------------------------------
            mstats = stats.get("memory_stats") or {}
            mem_stats_inner = mstats.get("stats") or {}
            mw = mem_stats_inner.get("active_anon", 0) or mstats.get("usage", 0)
            mem_working[lbl_vals] = mw
            mem_usage[lbl_vals] = mstats.get("usage", 0)

            # -- Network -----------------------------------------------------
            total_rx = total_tx = 0.0
            for iface, n in (stats.get("networks") or {}).items():
                rx = n.get("rx_bytes", 0)
                tx = n.get("tx_bytes", 0)
                total_rx += rx
                total_tx += tx
            net_rx[lbl_vals] = total_rx
            net_tx[lbl_vals] = total_tx

            # -- Disk --------------------------------------------------------
            total_r = total_w = 0.0
            for entry in stats.get("blkio_stats", {}).get("io_service_bytes_recursive") or []:
                op = entry.get("op", "").lower()
                v = entry.get("value", 0)
                if op == "read":
                    total_r += v
                elif op == "write":
                    total_w += v
            fs_read[lbl_vals] = total_r
            fs_write[lbl_vals] = total_w

        # ---- emit MetricFamily objects -----------------------------------

        # CPU  (Counter, with extra "cpu" label)
        cpu_lbl = base_lbl + ["cpu"]
        cpu_mf = CounterMetricFamily(
            "container_cpu_usage_seconds_total",
            "Cumulative cpu time consumed in seconds.",
            labels=cpu_lbl,
        )
        for key, val in cpu_usage.items():
            cpu_mf.add_metric(list(key), val)
        yield cpu_mf

        # Memory working set  (Gauge)
        mw_mf = GaugeMetricFamily(
            "container_memory_working_set_bytes",
            "Current working set in bytes.",
            labels=base_lbl,
        )
        for key, val in mem_working.items():
            mw_mf.add_metric(list(key), val)
        yield mw_mf

        # Memory usage  (Gauge)
        mu_mf = GaugeMetricFamily(
            "container_memory_usage_bytes",
            "Current memory usage in bytes.",
            labels=base_lbl,
        )
        for key, val in mem_usage.items():
            mu_mf.add_metric(list(key), val)
        yield mu_mf

        # Network receive  (Counter, aggregated across interfaces)
        rx_mf = CounterMetricFamily(
            "container_network_receive_bytes_total",
            "Cumulative count of bytes received.",
            labels=base_lbl,
        )
        for key, val in net_rx.items():
            rx_mf.add_metric(list(key), val)
        yield rx_mf

        # Network transmit  (Counter)
        tx_mf = CounterMetricFamily(
            "container_network_transmit_bytes_total",
            "Cumulative count of bytes transmitted.",
            labels=base_lbl,
        )
        for key, val in net_tx.items():
            tx_mf.add_metric(list(key), val)
        yield tx_mf

        # Disk reads  (Counter, aggregated across devices)
        dr_mf = CounterMetricFamily(
            "container_fs_reads_bytes_total",
            "Cumulative count of bytes read.",
            labels=base_lbl,
        )
        for key, val in fs_read.items():
            dr_mf.add_metric(list(key), val)
        yield dr_mf

        # Disk writes  (Counter)
        dw_mf = CounterMetricFamily(
            "container_fs_writes_bytes_total",
            "Cumulative count of bytes written.",
            labels=base_lbl,
        )
        for key, val in fs_write.items():
            dw_mf.add_metric(list(key), val)
        yield dw_mf

        # Last seen  (Gauge)
        now = time.time()
        ls_mf = GaugeMetricFamily(
            "container_last_seen",
            "Timestamp of the last time a container was seen.",
            labels=base_lbl,
        )
        # We didn't track per-container keys without stats, but we can
        # still emit for containers we saw above.
        seen_keys = set(mem_working.keys())
        for key in seen_keys:
            ls_mf.add_metric(list(key), now)
        yield ls_mf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    interval = float(os.environ.get("METRICS_POLL_INTERVAL_S", "10"))
    port = int(os.environ.get("METRICS_PORT", "9323"))

    collector = DockerCollector(interval=interval)
    REGISTRY.register(collector)

    start_http_server(port)
    log.info("Prometheus metrics server listening on :%d  (interval=%.1fs)", port, interval)

    # Block forever – the HTTP server runs in its own thread.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
