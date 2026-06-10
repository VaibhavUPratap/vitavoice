# Stage 1: Build the React application
FROM node:18-alpine AS build-stage

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: Serve the built application with Nginx
FROM nginx:stable-alpine

COPY --from=build-stage /app/dist /usr/share/nginx/html
# Copy custom nginx config if needed, otherwise use default
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
