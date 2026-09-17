FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS uv

# Install the project into /app
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings.
# `--extra cpu` is required, not cosmetic: torch lives in a mutually exclusive
# cpu/cu130 extra, and naming neither installs no torch at all. It also keeps
# 2.09 GB of CUDA runtime out of the image, which a container has no use for.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable --extra cpu --extra ocr-specialist --extra iqa

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra cpu --extra ocr-specialist --extra iqa

FROM python:3.13-slim-bookworm

WORKDIR /app

COPY --from=uv --chown=app:app /app/.venv /app/.venv

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

RUN huggingface-cli download florence-community/Florence-2-large

CMD ["fusion-vision-mcp", "--memory-mode", "persistent"]
