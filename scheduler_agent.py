import numpy as np
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from task_agent import TaskInt 
from task_queue_agent import RunableQueue
from scheduling_table import SchedulingTableInt


class Scheduler(object): 
    """
    Scheduler is responsible for the scheduling of the tasks:
    Task queue: 
    track ready tasks:
     (active): 
        Tasks are enqueued on some runqueue when they wake up and are dequeued when they are suspended.
        group processes into priority classes:  use priority scheduling among the classes but round-robin scheduling within each class
        track of deadlines of the earliest deadline tasks currently executing on each runqueue.
     (expired):
        Tasks are enqueued on some expired queue when they expire and are dequeued when they are refill. 
    track blocked tasks: 
        enqueue when they are blocked and are dequeued when they are unblocked.

    Preemption:
     condition: When a task is activated/increased priority on CPU k, which has higher priority than the executing one, 
     Operation: 
      a preemption happens, the preempted task is inserted at the head of the queue; 
      otherwise the wakenup task is inserted in the proper runqueue, depending on the state of the system. 

    Push: 
     condition: the head of the queue is modified, 
     operation: a push operation is executed to see if some task can be moved to another queue. 
    
    Pull:
     Condition: When a task suspends itself (due to blocking or sleeping) or lowers its priority on CPU k
     Operation: it looks at the other run-queues to see if some other higher priority tasks need to be migrated to the current CPU.

    1. maintain the scheduling table (SMT runable queue, for resources prevision) 
       enqueue new tasks, dequeue expired tasks, and adjust the position of the tasks in the runable queue
    2. maintain and monitor the task status: running, runnable, expired (throttled), suspended (blocked), terminated 
       update the task status when event happens 
       periodically (tile-level) checks whether it runs slower than expected due to resource contention
    3. maintain the event queue: 
        cache the pre-defined events (task release, task deadline, tick), 
        predict events (completion)
        record the runtime events (suspension, preemption, expiration, lag, spec update, delay, timeout) 
    4. make the scheduling decision:
        4.1. dispatch the task that can attain available resources
        4.2. make the preemption/pull/push decision
        4.3. enforce rule confinements

    track something: 
        number of Idle Cores by semaphore
        track of deadlines of the earliest deadline tasks currently executing on each runqueue.
    
    struct dl_rq {
        struct rb_root rb_root
        struct rb_node * rb_leftmost
        unsigned long dl_nr_running
        # ifdef CONFIG_SMP
        struct {
            / * two earliest tasks in queue * /
            u64 curr
            u64 next
            / * next earliest * /
        } earliest_dl
        int overloaded
        unsigned long dl_nr_migratory
        unsigned long dl_nr_total
        struct rb_root pushable_tasks_root
        struct rb_node * pushable_tasks_leftmost  # endif /* CONFIG_SMP */
    }
    • rb_root: the root of the red-black tree
    • rb_leftmost: the leftmost node of the red-black tree
    • dl_nr_running: the number of tasks in the run queue
    • earliest dl is a per-runqueue data structure used for “caching” the deadlines of the first two ready tasks, 
    so to facilitate migration-related decisions; 
    • dl_nr_migratory and dl_nr_total represent the number of queued tasks that can migrate and the total number of queued tasks, respectively; 
    • overloaded serves as a flag, and it is set when the queue contains more than one task; 
    • pushable_tasks_root is the root of the redblack tree of tasks that can be migrated, since they are queued but not running, 
    and it is ordered by increasing deadline; 
    • pushable_tasks_leftmost is a pointer to the node of pushable tasks root containing the task with the earliest deadline.
    """

    def __init__(self, allocator, num_resources, num_time_slots):
        # =============== 1. scheduling table ===============
        # scheduling table is a 2D array, each row is a resource, each column is a time slot
        # the value of each cell is the task id that occupies the resource in the time slot
        # the size of the scheduling table is determined by the number of resources and the number of time slots
        # the number of time slots is determined by the hyper-period of the tasks
        # the number of resources is determined by the number of resources that are available
        self.scheduling_table = SchedulingTableInt(num_resources, num_time_slots)
        # =============== 2. runable queue ===============
        # the runable queue is a queue that contains the tasks that are runnable
        # the runable queue is a priority queue that is sorted by the task deadline
        # the runable queue is used to make the scheduling decision
        self.runable_queue = RunableQueue()
        # =============== 3. event queue ===============
        # the event queue is a queue that contains the events that are generated by the task and the scheduler
        # the event queue is a priority queue that is sorted by the event time
        # the event queue is used to make the scheduling decision
        self.event_queue = OrderedDict()
        # =============== 4. task table ===============
        # the task table is a dictionary that contains the tasks that are managed by the scheduler
        # the task table is used to make the scheduling decision
        self.task_table = OrderedDict()
        # =============== 5. task id ===============
        # the task id is an integer that is used to identify the task
        # the task id is used to make the scheduling decision
        self.task_id = 0
        # =============== 6. hyper-period ===============
        # the hyper-period is an integer that is used to determine the size of the scheduling table
        # the hyper-period is used to make the scheduling decision
        self.hyper_period = 0
        # =============== 7. resource table ===============
        # the resource table is a dictionary that contains the resources that are managed by the scheduler
        # the resource table is used to make the scheduling decision
        self.resource_table = OrderedDict()
        
    def add_task(self, task:TaskInt, ) -> None:
        """
        add a task to the task table
        """
        self.task_table[task.id] = task
        self.runable_queue.put(task)
        self.hyper_period = np.lcm(self.hyper_period, task.period)
        self.scheduling_table = np.zeros((len(self.resource_table), self.hyper_period), dtype=int)
        self.event_queue = OrderedDict()
        self.task_id += 1
        

    