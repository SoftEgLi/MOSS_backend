// Package docs GENERATED BY SWAG; DO NOT EDIT
// This file was generated by swaggo/swag
package docs

import "github.com/swaggo/swag"

const docTemplate = `{
    "schemes": {{ marshal .Schemes }},
    "swagger": "2.0",
    "info": {
        "description": "{{escape .Description}}",
        "title": "{{.Title}}",
        "contact": {
            "name": "Maintainer Chen Ke",
            "url": "https://danxi.fduhole.com/about",
            "email": "dev@fduhole.com"
        },
        "license": {
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
        },
        "version": "{{.Version}}"
    },
    "host": "{{.Host}}",
    "basePath": "{{.BasePath}}",
    "paths": {
        "/": {
            "get": {
                "produces": [
                    "application/json"
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/models.Map"
                        }
                    }
                }
            }
        },
        "/chats": {
            "get": {
                "tags": [
                    "chat"
                ],
                "summary": "list user's chats",
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "type": "array",
                            "items": {
                                "$ref": "#/definitions/models.Chat"
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": [
                    "chat"
                ],
                "summary": "add a chat",
                "responses": {
                    "201": {
                        "description": "Created",
                        "schema": {
                            "$ref": "#/definitions/models.Chat"
                        }
                    }
                }
            }
        },
        "/chats/{chat_id}": {
            "put": {
                "tags": [
                    "chat"
                ],
                "summary": "modify a chat",
                "parameters": [
                    {
                        "type": "integer",
                        "description": "chat id",
                        "name": "chat_id",
                        "in": "path",
                        "required": true
                    },
                    {
                        "description": "json",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/chat.ModifyModel"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/models.Chat"
                        }
                    }
                }
            },
            "delete": {
                "tags": [
                    "chat"
                ],
                "summary": "delete a chat",
                "parameters": [
                    {
                        "type": "integer",
                        "description": "chat id",
                        "name": "chat_id",
                        "in": "path",
                        "required": true
                    }
                ],
                "responses": {
                    "204": {
                        "description": "No Content"
                    }
                }
            }
        },
        "/chats/{chat_id}/records": {
            "get": {
                "tags": [
                    "record"
                ],
                "summary": "list records of a chat",
                "parameters": [
                    {
                        "type": "integer",
                        "description": "chat id",
                        "name": "chat_id",
                        "in": "path",
                        "required": true
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "type": "array",
                            "items": {
                                "$ref": "#/definitions/models.Record"
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": [
                    "record"
                ],
                "summary": "add a record",
                "parameters": [
                    {
                        "type": "integer",
                        "description": "chat id",
                        "name": "chat_id",
                        "in": "path",
                        "required": true
                    },
                    {
                        "description": "json",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/record.CreateModel"
                        }
                    }
                ],
                "responses": {
                    "201": {
                        "description": "Created",
                        "schema": {
                            "$ref": "#/definitions/models.Record"
                        }
                    }
                }
            }
        },
        "/chats/{chat_id}/regenerate": {
            "put": {
                "tags": [
                    "record"
                ],
                "summary": "regenerate the last record of a chat",
                "parameters": [
                    {
                        "type": "integer",
                        "description": "chat id",
                        "name": "chat_id",
                        "in": "path",
                        "required": true
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/models.Record"
                        }
                    }
                }
            }
        },
        "/config": {
            "get": {
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "Config"
                ],
                "summary": "get global config",
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/config.Response"
                        }
                    }
                }
            }
        },
        "/login": {
            "post": {
                "description": "Login with email and password, return jwt token, not need jwt",
                "consumes": [
                    "application/json"
                ],
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "token"
                ],
                "summary": "Login",
                "parameters": [
                    {
                        "description": "json",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/account.LoginRequest"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/account.TokenResponse"
                        }
                    },
                    "400": {
                        "description": "Bad Request",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    },
                    "404": {
                        "description": "User Not Found",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    },
                    "500": {
                        "description": "Internal Server Error",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            }
        },
        "/logout": {
            "get": {
                "description": "Logout, clear jwt credential and return successful message, logout, login required",
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "token"
                ],
                "summary": "Logout",
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            }
        },
        "/records/{record_id}": {
            "put": {
                "tags": [
                    "record"
                ],
                "summary": "modify a record",
                "parameters": [
                    {
                        "type": "integer",
                        "description": "record id",
                        "name": "record_id",
                        "in": "path",
                        "required": true
                    },
                    {
                        "description": "json",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/record.ModifyModel"
                        }
                    }
                ],
                "responses": {
                    "201": {
                        "description": "Created",
                        "schema": {
                            "$ref": "#/definitions/models.Record"
                        }
                    }
                }
            }
        },
        "/refresh": {
            "post": {
                "description": "Refresh jwt token with refresh token in header, login required",
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "token"
                ],
                "summary": "Refresh jwt token",
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/account.TokenResponse"
                        }
                    }
                }
            }
        },
        "/register": {
            "put": {
                "description": "reset password, reset jwt credential",
                "consumes": [
                    "application/json"
                ],
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "account"
                ],
                "summary": "reset password",
                "parameters": [
                    {
                        "description": "json",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/account.RegisterRequest"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/account.TokenResponse"
                        }
                    },
                    "400": {
                        "description": "验证码错误",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    },
                    "500": {
                        "description": "Internal Server Error",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            },
            "post": {
                "description": "register with email or phone, password and verification code",
                "consumes": [
                    "application/json"
                ],
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "account"
                ],
                "summary": "register",
                "parameters": [
                    {
                        "description": "json",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/account.RegisterRequest"
                        }
                    }
                ],
                "responses": {
                    "201": {
                        "description": "Created",
                        "schema": {
                            "$ref": "#/definitions/account.TokenResponse"
                        }
                    },
                    "400": {
                        "description": "验证码错误、用户已注册",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    },
                    "500": {
                        "description": "Internal Server Error",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            }
        },
        "/users/me": {
            "get": {
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "user"
                ],
                "summary": "get current user",
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/models.User"
                        }
                    },
                    "404": {
                        "description": "User not found",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    },
                    "500": {
                        "description": "Internal Server Error",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            },
            "put": {
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "user"
                ],
                "summary": "modify user, need login",
                "parameters": [
                    {
                        "description": "json",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/account.ModifyUserRequest"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/models.User"
                        }
                    },
                    "500": {
                        "description": "Internal Server Error",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            },
            "delete": {
                "description": "delete user and related jwt credentials",
                "tags": [
                    "account"
                ],
                "summary": "delete user",
                "parameters": [
                    {
                        "description": "email, password",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/account.LoginRequest"
                        }
                    }
                ],
                "responses": {
                    "204": {
                        "description": "No Content"
                    },
                    "400": {
                        "description": "密码错误“",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    },
                    "404": {
                        "description": "用户不存在“",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    },
                    "500": {
                        "description": "Internal Server Error",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            }
        },
        "/verify/email": {
            "get": {
                "description": "verify with email in query, Send verification email",
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "account"
                ],
                "summary": "verify with email in query",
                "parameters": [
                    {
                        "type": "string",
                        "name": "email",
                        "in": "query"
                    },
                    {
                        "enum": [
                            "register",
                            "reset",
                            "modify"
                        ],
                        "type": "string",
                        "name": "scope",
                        "in": "query"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/account.VerifyResponse"
                        }
                    },
                    "400": {
                        "description": "已注册“",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            }
        },
        "/verify/phone": {
            "get": {
                "description": "verify with phone in query, Send verification message",
                "produces": [
                    "application/json"
                ],
                "tags": [
                    "account"
                ],
                "summary": "verify with phone in query",
                "parameters": [
                    {
                        "type": "string",
                        "description": "phone number in e164 mode",
                        "name": "phone",
                        "in": "query"
                    },
                    {
                        "enum": [
                            "register",
                            "reset",
                            "modify"
                        ],
                        "type": "string",
                        "name": "scope",
                        "in": "query"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {
                            "$ref": "#/definitions/account.VerifyResponse"
                        }
                    },
                    "400": {
                        "description": "已注册“",
                        "schema": {
                            "$ref": "#/definitions/utils.MessageResponse"
                        }
                    }
                }
            }
        },
        "/ws/chats/{chat_id}/records": {
            "get": {
                "tags": [
                    "Websocket"
                ],
                "summary": "add a record",
                "parameters": [
                    {
                        "type": "integer",
                        "description": "chat id",
                        "name": "chat_id",
                        "in": "path",
                        "required": true
                    },
                    {
                        "description": "json",
                        "name": "json",
                        "in": "body",
                        "required": true,
                        "schema": {
                            "$ref": "#/definitions/record.CreateModel"
                        }
                    }
                ],
                "responses": {
                    "201": {
                        "description": "Created",
                        "schema": {
                            "$ref": "#/definitions/models.Record"
                        }
                    }
                }
            }
        }
    },
    "definitions": {
        "account.LoginRequest": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string"
                },
                "password": {
                    "type": "string",
                    "minLength": 8
                },
                "phone": {
                    "description": "phone number in e164 mode",
                    "type": "string"
                }
            }
        },
        "account.ModifyUserRequest": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string"
                },
                "nickname": {
                    "type": "string",
                    "minLength": 1
                },
                "phone": {
                    "description": "phone number in e164 mode",
                    "type": "string"
                },
                "share_consent": {
                    "type": "boolean"
                },
                "verification": {
                    "type": "string",
                    "maxLength": 6,
                    "minLength": 6
                }
            }
        },
        "account.RegisterRequest": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string"
                },
                "invite_code": {
                    "type": "string",
                    "minLength": 1
                },
                "password": {
                    "type": "string",
                    "minLength": 8
                },
                "phone": {
                    "description": "phone number in e164 mode",
                    "type": "string"
                },
                "verification": {
                    "type": "string",
                    "maxLength": 6,
                    "minLength": 6
                }
            }
        },
        "account.TokenResponse": {
            "type": "object",
            "properties": {
                "access": {
                    "type": "string"
                },
                "message": {
                    "type": "string"
                },
                "refresh": {
                    "type": "string"
                }
            }
        },
        "account.VerifyResponse": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string"
                },
                "scope": {
                    "type": "string",
                    "enum": [
                        "register",
                        "reset"
                    ]
                }
            }
        },
        "chat.ModifyModel": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1
                }
            }
        },
        "config.Response": {
            "type": "object",
            "properties": {
                "invite_required": {
                    "type": "boolean"
                },
                "region": {
                    "type": "string"
                }
            }
        },
        "models.Chat": {
            "type": "object",
            "properties": {
                "count": {
                    "description": "Record 条数",
                    "type": "integer"
                },
                "created_at": {
                    "type": "string"
                },
                "id": {
                    "type": "integer"
                },
                "name": {
                    "type": "string"
                },
                "records": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/models.Record"
                    }
                },
                "updated_at": {
                    "type": "string"
                },
                "user_id": {
                    "type": "integer"
                }
            }
        },
        "models.Map": {
            "type": "object",
            "additionalProperties": {}
        },
        "models.Record": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer"
                },
                "created_at": {
                    "type": "string"
                },
                "duration": {
                    "description": "处理时间，单位 s",
                    "type": "number"
                },
                "feedback": {
                    "type": "string"
                },
                "id": {
                    "type": "integer"
                },
                "like_data": {
                    "description": "1 like, -1 dislike",
                    "type": "integer"
                },
                "request": {
                    "type": "string"
                },
                "request_sensitive": {
                    "type": "boolean"
                },
                "response": {
                    "type": "string"
                },
                "response_sensitive": {
                    "type": "boolean"
                }
            }
        },
        "models.User": {
            "type": "object",
            "properties": {
                "chats": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/models.Chat"
                    }
                },
                "email": {
                    "type": "string"
                },
                "id": {
                    "type": "integer"
                },
                "joined_time": {
                    "type": "string"
                },
                "last_login": {
                    "type": "string"
                },
                "nickname": {
                    "type": "string"
                },
                "phone": {
                    "type": "string"
                },
                "share_consent": {
                    "type": "boolean"
                }
            }
        },
        "record.CreateModel": {
            "type": "object",
            "required": [
                "request"
            ],
            "properties": {
                "request": {
                    "type": "string"
                }
            }
        },
        "record.ModifyModel": {
            "type": "object",
            "properties": {
                "feedback": {
                    "type": "string"
                },
                "like": {
                    "description": "1 like, -1 dislike, 0 reset",
                    "type": "integer",
                    "enum": [
                        1,
                        0,
                        -1
                    ]
                }
            }
        },
        "utils.MessageResponse": {
            "type": "object",
            "properties": {
                "data": {},
                "message": {
                    "type": "string"
                }
            }
        }
    }
}`

// SwaggerInfo holds exported Swagger Info so clients can modify it
var SwaggerInfo = &swag.Spec{
	Version:          "0.0.1",
	Host:             "localhost:8000",
	BasePath:         "/api",
	Schemes:          []string{},
	Title:            "Moss Backend",
	Description:      "Moss Backend",
	InfoInstanceName: "swagger",
	SwaggerTemplate:  docTemplate,
}

func init() {
	swag.Register(SwaggerInfo.InstanceName(), SwaggerInfo)
}
