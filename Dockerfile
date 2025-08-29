# ===== Stage 1: Builder =====
# This stage installs dependencies in an isolated environment.
FROM python:3.12 AS builder

# 1. Install Poetry
# Used to manage the project dependencies.
RUN pip install poetry

# 2. Configure Poetry environment
# Disables the creation of virtual environments, since we’ll handle the environment manually.
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
WORKDIR /app

# 3. Copy dependency files and install them
# We copy only these files first to take advantage of Docker’s cache.
# This layer is only rebuilt if dependencies change.
COPY pyproject.toml ./
RUN poetry install --only=main --no-root


# ===== Stage 2: Runner =====
# This is the final stage that creates the production image.
FROM python:3.12-slim

# 1. Create a non-root user
# Running as non-root is a fundamental security practice.
RUN useradd --create-home appuser
WORKDIR /home/appuser

# 2. Copy dependencies from the 'builder' stage
# We copy only the virtual environment with the already installed libraries.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 3. Copy artifacts and source code
COPY artifacts/ ./artifacts/
COPY src/ ./src/

# 4. Assign permissions to the non-root user
RUN chown -R appuser:appuser /home/appuser
USER appuser

# 5. Expose the port
# Port 8000 is the standard used by Uvicorn.
EXPOSE 8000

# 6. Define the entrypoint to run the application
# Starts the Uvicorn server so the API is accessible.
ENTRYPOINT ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]


# Using a builder stage and a runner stage is a technique called multi-stage build.
# It is not mandatory, but it is a best practice recommended by professionals for creating production images.

# Builder stage
# Purpose: A temporary and disposable build environment.
# Contents: Includes everything needed to compile and prepare your app:
#   compilers (like gcc), build tools (poetry), and full versions of libraries.
# Outcome: Once it installs and compiles everything, its only job is to serve
#   as a source for the next stage to copy the ready-made files.

# Runner stage
# Purpose: The final, optimized, lightweight image used to run your app.
# Contents: Contains only what is strictly necessary for the app to work:
#   the Python interpreter, your source code, and the pre-installed libraries
#   copied from the builder stage.
# Outcome: A small, secure, production-ready image.

# Why use both?
# Image Size: The final image is much smaller. The builder stage can weigh several GB
#   with all tools, but the runner is only a few MB. This speeds up downloads and deployments.
# Security: The runner image has a much smaller attack surface.
#   It doesn’t include compilers, poetry, or other tools an attacker could exploit
#   if they got into the container.
# In summary: you use a “big and dirty” stage (builder) to do all the heavy installation work,
# then copy only the clean results to a final “small and secure” stage (runner).
