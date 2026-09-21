"""Consistent JSON bodies for success and error responses."""

from flask import jsonify


def success(data: dict, status: int = 200):
    return jsonify(data), status


def error(code: str, message: str, details=None, status: int = 400):
    body = {"error": code, "message": message}
    if details is not None:
        body["details"] = details
    return jsonify(body), status
