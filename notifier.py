from dotenv import dotenv_values
import numpy as np
import notifiers
import keras
import keras.callbacks


secrets = dotenv_values("telegram.env")
TOKEN = secrets['TOKEN']
CHAT_ID = int(secrets['CHAT_ID'])


class TelegramNotifier():
    def __init__(self, prefix=None, token=TOKEN, chat_id=CHAT_ID):
        self._telegram = notifiers.get_notifier('telegram')
        self._token = token
        self._chat_id = chat_id
        self._prefix = prefix

    def send_message(self, text, also_print=True):
        if also_print:
            print(text)
        msg = f"**{self._prefix}:**\n{text}" if self._prefix else text
        self._telegram.notify(message=msg, token=self._token, chat_id=self._chat_id)


class Notify(keras.callbacks.Callback):
    def __init__(self, num_epochs, prefix=None):
        super(Notify, self).__init__()
        self._notifier = TelegramNotifier(prefix)
        self._message = "Epoch {}/{}\nLoss: {:.3}\nAcc: {:.3%}\nVal.Loss: {:.3}\nVal.Acc: {:.3%}"
        self._num_epochs = num_epochs

    def on_epoch_end(self, epoch, logs=None):
        try:
            msg = self._message.format(
                epoch + 1,
                self._num_epochs,
                logs.get('loss'),
                logs.get('binary_accuracy'),
                logs.get('val_loss'),
                logs.get('val_binary_accuracy'),
            )
            self._notifier.send_message(msg)
        except:
            self._notifier.send_message(f"Epoch {epoch}/{self._num_epochs}\n{logs}")
