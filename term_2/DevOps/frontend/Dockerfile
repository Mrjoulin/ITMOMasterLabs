# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Declare build arguments
ARG VITE_API_URL

# Copy source code
COPY . ./

# Set environment variable and build
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# Stage 2: Production with nginx
FROM nginx:alpine AS runtime

# Install curl for health checks
RUN apk add --no-cache curl

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built app from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
