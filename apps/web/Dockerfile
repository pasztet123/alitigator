FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_API_BASE_URL=https://alitigator-api-498660485375.europe-north2.run.app
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM node:20-alpine AS runtime

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=5173

RUN npm install -g serve

COPY --from=build /app/dist ./dist

EXPOSE 5173

CMD ["sh", "-c", "serve -s dist -l tcp://0.0.0.0:${PORT:-5173}"]
