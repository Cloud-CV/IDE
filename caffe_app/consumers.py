from channels import Group # noqa: ignore=F405
from channels.auth import channel_session_user, channel_session_user_from_http


@channel_session_user_from_http
def ws_connect(message):
    pass


@channel_session_user
def ws_disconnect(message):
    pass


@channel_session_user
def ws_receive(message):
    pass
