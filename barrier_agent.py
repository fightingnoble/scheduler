# simulate a barrier 
# the barrier resets after a certain time
class Barrier(object):
    def __init__(self, reset_time):
        self.number_of_asserts = 0
        self.assert_barrier(reset_time)
        
    
    def update(self, delta_time):
        self.reset_timer -= delta_time
        if self.reset_timer < 0:
            self.reset_timer = 0
        return self.reset_timer > 0
        
    def assert_barrier(self, reset_time):
        self.reset_time = reset_time
        self.reset_timer = reset_time
        self.number_of_asserts += 1
    
    def state(self):
        return self.reset_timer > 0
    
# test code for Barrier class
def test_barrier():
    # test initialization
    b = Barrier(10)
    assert b.state() == True
    
    # test update
    assert b.update(5) == True
    assert b.state() == True
    assert b.update(5) == False
    assert b.state() == False
    
    # test reset
    b.assert_barrier(5)
    assert b.state() == True
    assert b.update(4) == True
    assert b.state() == True
    assert b.update(1) == False
    assert b.state() == False

# test_barrier()
if __name__ == "__main__":
    test_barrier()