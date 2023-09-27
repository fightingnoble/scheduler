class PosTableInt(object): 
    def __init__(self, num_records) -> None:
        self.records = [dict() for i in range(num_records)]
        self.num_records = num_records
        self.idx_table = dict()
    
    def get_record(self, key):
        return self.records[self.idx_table[key]]
    
    def set_record(self, key, value):
        if key not in self.idx_table:
            self.idx_table[key] = len(self.idx_table)
        self.records[self.idx_table[key]] = value