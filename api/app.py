# In api/app.py, update the CORS configuration:

from api.middleware import create_cors_middleware, get_cors_config_from_env

# Get CORS configuration
cors_config = get_cors_config_from_env()

# Add CORS middleware
app.add_middleware(
    create_cors_middleware,
    allow_origins=cors_config["allow_origins"],
    allow_origin_regex=cors_config["allow_origin_regex"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
    allow_credentials=cors_config["allow_credentials"],
    expose_headers=cors_config["expose_headers"],
    max_age=cors_config["max_age"],
    allow_private_network=cors_config["allow_private_network"],
    enhanced=True  # Use enhanced CORS middleware with logging
)
