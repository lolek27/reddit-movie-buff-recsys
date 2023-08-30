# from pysondb import db
from tinydb import TinyDB, Query

from typing import Final
from pydantic import BaseModel


USER_QUERY: Final[int] = 1
USER_ANSWERS: Final[int] = 2
USER_FEEDBACK: Final[int] = 3


class StoreValue:
    type: int
    value: dict


Entry = Query()


class AppStore():
    def __init__(self, path='.') -> None:
        self.db_path = path + '/db.json'
        self.store = TinyDB(self.db_path)

    def addToStore(self, entry: StoreValue):
        self.store.insert(entry)

    def getAll(self):
        return self.store.all()

    def getByQueryByType(self, query_fn):
        return self.store.search(Entry.type.test(query_fn))

    def getByQueryByValue(self, query_fn):
        return self.store.search(Entry.value.test(query_fn))
