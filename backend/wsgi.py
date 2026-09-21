"""WSGI entry: gunicorn wsgi:app — optional for later deployment."""

from app import create_app

app = create_app()
