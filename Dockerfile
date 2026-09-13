# ===========================================================================
# Stage 1: Build Stage (Official Rust Debian Slim Image)
# ===========================================================================
FROM rust:1.98.1-slim-bookworm AS builder

# Bust any stale Railway builder cache
ARG CACHEBUST=2026091401

WORKDIR /app

# Install build dependencies and verify toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && echo "=== Active Rust Toolchain ===" && rustc --version

# Copy workspace manifests
COPY Cargo.toml ./
COPY cmd/server/Cargo.toml cmd/server/Cargo.toml
COPY libs/document/Cargo.toml libs/document/Cargo.toml
COPY libs/orm/Cargo.toml libs/orm/Cargo.toml
COPY migration/Cargo.toml migration/Cargo.toml

# Copy source code
COPY cmd/ cmd/
COPY libs/ libs/
COPY migration/ migration/

# Build release binaries
RUN cargo build --release --package server --package migration

# ===========================================================================
# Stage 2: Minimal Runtime Image (~80 MB, ultra-fast & low memory)
# ===========================================================================
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime SSL certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled binaries from builder
COPY --from=builder /app/target/release/server /app/server
COPY --from=builder /app/target/release/migration /app/migration

# Default environment settings
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000

EXPOSE 8000

# Run migrations and launch the Axum server
CMD ["sh", "-c", "/app/migration up && /app/server"]
