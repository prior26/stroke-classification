import threading
import requests
import queue

class MyLogger:
    def __init__(self, server_url:str = None, server_username:str = None, server_folder:str = None, model_name:str = None, path_localFile:str = None):
        self.SERVER_URL = server_url
        self.SERVER_USERNAME = server_username
        self.SERVER_FOLDER = server_folder
        self.MODEL_NAME = model_name
        self.PATH_LOCAL_LOG_FILE = path_localFile
        self.IS_SERVER_WORKING = True
        
        self.local_queue = queue.Queue()
        self.server_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.local_thread = threading.Thread(target=self._local_writer_thread, daemon=True)
        self.server_thread = threading.Thread(target=self._server_writer_thread, daemon=True)
        if path_localFile:
            self.local_thread.start()
            print("Started Local Writer Thread")
        else:
            print("Local writer thread not started. No local file path provided.")

        if server_url:
            self.server_thread.start()
            print("Started Server Writer Thread")
        else:
            print("Server writer thread not started. No server URL provided")
        
    def _local_writer_thread(self):
        while not self.stop_event.is_set() or not self.local_queue.empty():
            try:
                string = self.local_queue.get(timeout=2)
                with open(self.PATH_LOCAL_LOG_FILE, "a") as log_file:
                    log_file.write(string + "\n")
                self.local_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Local Writer Thread Exception: {type(e).__name__}: {e}")
    
    def _server_writer_thread(self):
        while not self.stop_event.is_set() or not self.server_queue.empty():
            try:
                logs = []
                while not self.server_queue.empty():
                    logs.append(self.server_queue.get_nowait())
                    self.server_queue.task_done()
                if(len(logs) > 0):
                    data = {
                        "msg": "\n".join(logs),
                        "main_folder": self.SERVER_FOLDER,
                        "model_name": self.MODEL_NAME,
                        "user_name": self.SERVER_USERNAME
                    }
                    try:
                        response = requests.post(url=self.SERVER_URL, data=data)
                        if response.status_code != 200:
                            if(not self.IS_SERVER_WORKING):
                                print(f"Error Writing on Monitor Server: {response.status_code}")
                                print(f"Response: {response.text}")
                            self.IS_SERVER_WORKING = False
                        else:
                            self.IS_SERVER_WORKING = True
                    
                    except Exception as e:
                        if(self.IS_SERVER_WORKING):
                            print(f"Error while logging on server: {e}")
                            self.IS_SERVER_WORKING = False
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Server Writer Unexpected Exception: {type(e).__name__}: {e}")

    def log(self, string):
        # print to console
        print(string)
        
        # write to local file
        if(self.PATH_LOCAL_LOG_FILE is not None and self.PATH_LOCAL_LOG_FILE != ""):
            self.local_queue.put(string)
        
        # write on server
        if self.SERVER_URL is not None and self.SERVER_URL != "":
            self.server_queue.put(string)

    def stop(self):
        self.local_queue.join()
        self.server_queue.join()
        self.stop_event.set()
        if self.PATH_LOCAL_LOG_FILE: self.local_thread.join()
        if self.SERVER_URL: self.server_thread.join()
