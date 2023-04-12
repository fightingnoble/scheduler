# A message classmethod 
# Function: a message caching, sending and receiving data class
# Author: Chenguang Zhang
# Date: 2023-03-19

class MessagePipe:
    cache = []

    @classmethod
    def send(cls, data, prefix=""):
        cls.cache.append(data)
        print(prefix+f"Sending message: {data}")

    @classmethod
    def receive(cls):
        if cls.cache:
            data = cls.cache.pop(0)
            print(f"Receiving message: {data}")
            return data
        else:
            print("No messages to receive.")
    
    @classmethod
    def clear(cls):
        cls.cache = []
    
    @classmethod
    def get_cache(cls):
        return cls.cache
    
    @classmethod
    # filter the message by the keyword
    def filter(cls, keyword):
        return [msg for msg in cls.cache if keyword in msg]
    
    def empty(cls):
        return len(cls.cache) == 0


# Test the class Message with the example of multiple instance communication
if __name__ == "__main__":
    # Create two instances of Message
    msg1 = MessagePipe()
    msg2 = MessagePipe()

    # Send messages from msg1 to msg2
    msg1.send("Hello from msg1")
    msg1.send("How are you doing?")

    # Receive messages in msg2
    msg2.receive()
    msg2.receive()

    # Send a message from msg2 to msg1
    msg2.send("Hi, I\'m doing well. Thanks for asking.")

    # Receive the message in msg1
    msg1.receive()

    # Try to receive another message in msg1 (should print "No messages to receive.")
    msg1.receive()

