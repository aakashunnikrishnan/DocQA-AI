# DocQA AI API Documentation

**Version:** 1.0.0
**Base URL:** `https://api.docqa-ai.com/v1` (production) or `http://localhost:8000/api/v1` (development)

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Endpoints](#endpoints)
  - [Query API](#query-api)
  - [Document Management](#document-management)
  - [Task Management](#task-management)
  - [System](#system)
- [WebSocket API](#websocket-api)
- [Error Handling](#error-handling)
- [SDKs and Libraries](#sdks-and-libraries)
- [FAQ](#faq)

---

## Authentication

The API uses API keys for authentication. Include your API key in the `Authorization` header:

```http
Authorization: Bearer YOUR_API_KEY
