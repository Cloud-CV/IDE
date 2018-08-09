from channels.routing import route, include
from channels.staticfiles import StaticFilesConsumer # noqa: ignore=F405
from caffe_app.consumers import ws_connect, ws_disconnect, ws_receive
from caffe_app.consumers import celery_response_handler

# routes defined for channel calls
# this is similar to the Django urls, but specifically for Channels
ws_routing = [
    route('websocket.connect', ws_connect),
    route('websocket.receive', ws_receive),
    route('websocket.disconnect', ws_disconnect),
    route('celery-response', celery_response_handler),
]

channel_routing = [
    include(ws_routing, path=r"^/ws/connect"),
]
