FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_API_BASE_URL=https://alitigator-api-498660485375.europe-north2.run.app
ARG VITE_SUPABASE_URL=https://opetgkmjhvrkufyewibr.supabase.co
ARG VITE_SUPABASE_PUBLISHABLE_KEY=sb_publishable_E3KG7MTfQG7gwoY5HkYAVg_WbawoUTg
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_SUPABASE_URL=${VITE_SUPABASE_URL}
ENV VITE_SUPABASE_PUBLISHABLE_KEY=${VITE_SUPABASE_PUBLISHABLE_KEY}

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
