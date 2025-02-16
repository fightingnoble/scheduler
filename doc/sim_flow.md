check complete and missing at the begin of each scheduling step





record the start time of newly issued task

update the information of the running task


 a task is initialized in multiple queues, and we need to ensure that the remburst is only released once, 
    so we need to check if the remburst is 0 before adding the totcpu to it: 
```
        if self.remburst == 0:
            self.remburst += self.totcpu
```