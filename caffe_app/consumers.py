import json
import yaml
import urlparse
from channels import Group
from channels.auth import channel_session_user, channel_session_user_from_http


@channel_session_user_from_http
def ws_connect(message):
    print('connection being established...')
    message.reply_channel.send({
        'accept': True
    })
    params = urlparse.parse_qs(message.content['query_string'])
    networkId = params.get('id',('Not Supplied',))[0]
    message.channel_session['networkId'] = networkId
    print('model-{0}'.format(networkId))
    Group('model-{0}'.format(networkId)).add(message.reply_channel)



@channel_session_user
def ws_disconnect(message):
    networkId = message.channel_session['networkId']
    Group('model-{0}'.format(networkId)).discard(message.reply_channel)
    print('disconnected...')


@channel_session_user
def ws_receive(message):
    print('message received...')
    data = yaml.safe_load(message['text'])
    networkId = message.channel_session['networkId']
    net = data['net']
    Group('model-{0}'.format(networkId)).send({
        'text': json.dumps({
            'net': net
        })
    })
