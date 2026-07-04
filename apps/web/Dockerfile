FROM node:20-alpine AS build

WORKDIR /app/apps/web

COPY apps/web/package*.json ./
RUN npm ci

COPY apps/web ./
RUN npm run build

FROM node:20-alpine AS runtime

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=5173

RUN npm install -g serve

COPY --from=build /app/apps/web/dist ./dist

EXPOSE 5173

CMD ["sh", "-c", "serve -s dist -l tcp://0.0.0.0:${PORT:-5173}"]
