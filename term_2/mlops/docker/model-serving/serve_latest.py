import glob
import os
import shutil
import sys
import tempfile
import time
import signal
import subprocess
from mlflow import MlflowClient

# --- Configuration from environment ---
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:4444")
MODEL_NAME = os.getenv("MODEL_SERVE_NAME")
MODEL_HOST = os.getenv("MODEL_SERVE_HOST", "0.0.0.0")
MODEL_PORT = os.getenv("MODEL_SERVE_PORT", "3000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 5))
MLFLOW_TMP_GLOB = os.getenv(
    "MLFLOW_TMP_GLOB", os.path.join(tempfile.gettempdir(), "tmp*")
)

if not MODEL_NAME:
    sys.exit("ERROR: MODEL_SERVE_NAME environment variable not set")


def get_latest_version(client, model_name):
    """Return the latest version number for the given registered model."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    # Versions are returned in descending order by default, so the first is latest
    return max(int(v.version) for v in versions)


def serve_model(model_name, version):
    """Start mlflow models serve as a subprocess and return the Popen object."""
    model_uri = f"models:/{model_name}/{version}"
    cmd = [
        "mlflow", "models", "serve",
        "--model-uri", model_uri,
        "--host", MODEL_HOST,
        "--port", MODEL_PORT,
        "--env-manager", "uv"
    ]
    print(f"Starting server for {model_uri}...")
    return subprocess.Popen(cmd)


def cleanup_old_model_cache():
    """Remove leftover MLflow temp artifact dirs from previous serve subprocesses.

    Safe to call between stopping the old server and starting the new one:
    the old server's temp dir is no longer in use, and the new server has
    not started yet, so no live server's cache is touched.
    """
    removed = 0
    for path in glob.glob(MLFLOW_TMP_GLOB):
        try:
            shutil.rmtree(path)
            removed += 1
            print(f"Removed cached model artifacts: {path}")
        except Exception as e:
            print(f"Warning: could not remove {path}: {e}")
    if removed:
        print(f"Cache cleanup done: removed {removed} director(y/es).")


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    current_version = None
    server_process = None

    while True:
        try:
            latest_version = get_latest_version(client, MODEL_NAME)
        except Exception as e:
            print(f"Error querying model registry: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        if latest_version is not None and latest_version != current_version:
            print(f"New version detected: {latest_version} (was {current_version})")
            # Stop the old server if running
            if server_process and server_process.poll() is None:
                print("Stopping old server...")
                server_process.send_signal(signal.SIGTERM)
                try:
                    server_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    server_process.wait()
            cleanup_old_model_cache()
            # Start the new server
            server_process = serve_model(MODEL_NAME, latest_version)
            current_version = latest_version

        # Check if the server is still alive (crashed / killed externally)
        if server_process and server_process.poll() is not None:
            print("Server process exited unexpectedly, restarting...")
            server_process = serve_model(MODEL_NAME, current_version)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
