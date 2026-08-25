FROM node:22-alpine AS web
WORKDIR /build
RUN corepack enable
COPY bti/frontend/package.json bti/frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile --node-linker=hoisted --package-import-method=copy
COPY bti/frontend/ ./
RUN pnpm build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BTI_HOST=0.0.0.0 PORT=8000
WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY bti/server/requirements.txt ./bti-server-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r bti-server-requirements.txt
COPY . .
COPY --from=web /build/dist ./bti/frontend/dist
EXPOSE 8000
CMD ["python", "-m", "bti.server"]
