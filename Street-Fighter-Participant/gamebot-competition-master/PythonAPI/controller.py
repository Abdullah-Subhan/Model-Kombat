import socket
import json
import sys
import time
from game_state import GameState
from command import Command
from buttons import Buttons
from listen_to_key import get_current_keypress
from make_dataset import record_frame

player_id = sys.argv[1]
MODE = 'record' if len(sys.argv) > 2 and sys.argv[2] == 'record' else 'bot'
port = 9999 if player_id == '1' else 10000

def connect(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', port))
    s.listen(1)
    print(f"[Controller] Waiting for game connection on port {port} (Player {player_id})...")
    client, _ = s.accept()
    print('[Controller] Connected to game!')
    return client

def send(sock, cmd):
    payload = json.dumps(cmd.object_to_dict())
    sock.sendall(payload.encode())

def receive(sock):
    data = json.loads(sock.recv(4096).decode())
    return GameState(data)

def main():
    sock = connect(port)
    cmd = Command()
    bot = None
    
    try:
        while True:
            gs = receive(sock)
        
            if MODE != 'record' and bot is None:
                from bot import Bot
                our_char_id = gs.player1.player_id if player_id == '1' else gs.player2.player_id
                print(f"[Controller] Launching rule-compliant Bot for Player {player_id} (Initial Character ID: {our_char_id})...")
                bot = Bot(player_id=our_char_id)

            if MODE == 'record':
                keys = get_current_keypress()
                cmd.player_buttons = Buttons({k: True for k in keys})
                record_frame(gs, keys)
            else:
                cmd = bot.fight(gs, player_id)
            
            send(sock, cmd)
            time.sleep(1/60.0)
    except KeyboardInterrupt:
        print("\n[Controller] Stopping controller...")
    finally:
        if bot and hasattr(bot, 'logger') and bot.logger:
            bot.logger.close()
        sock.close()

if __name__ == '__main__':
    main()
