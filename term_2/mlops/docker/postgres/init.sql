-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schema for Polymarket data
CREATE SCHEMA IF NOT EXISTS polymarket;

-- Create database for Mlflow
CREATE DATABASE mlflow;
