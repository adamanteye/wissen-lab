FROM node:25-alpine AS builder
RUN npm install -g pnpm@11.22.0
WORKDIR /srv
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

FROM nginx:alpine AS runtime
COPY --from=builder /srv/dist /usr/share/nginx/html
