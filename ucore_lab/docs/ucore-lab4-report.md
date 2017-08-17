## 内核线程管理

#### 内核线程概述

* 内核线程
	- 内核线程是一种特殊的进程
	- 内核线程只运行在内核态，而用户进程会在用户态和内核态交替运行
	- 所有内核线程直接使用共同的ucore内核空间；用户进程需要维护各自的用户内存空间 


* 进程控制块
	- 为了实现内核线程，需要设计管理线程的数据结构，即进程控制块（线程控制块）
	- 首先要创建内核线程对应的进程控制块，还需要把这些控制块通过链表连在一起，便于随时进行插入，删除和操作等进程管理操作
	- 通过调度器让不同的内核线程在不同的时间占用CPU执行，实现对CPU的分时共享
	

 * 进程控制块数据结构
	- 进程管理信息用struct proc_struct表示
	```
	struct proc_struct {
	    enum proc_state state;                      // Process state
	    int pid;                                    // Process ID
	    int runs;                                   // the running times of Proces
	    uintptr_t kstack;                           // Process kernel stack
	    volatile bool need_resched;                 // bool value: need to be rescheduled to release CPU?
	    struct proc_struct *parent;                 // the parent process
	    struct mm_struct *mm;                       // Process's memory management field
	    struct context context;                     // Switch here to run process
	    struct trapframe *tf;                       // Trap frame for current interrupt
	    uintptr_t cr3;                              // CR3 register: the base addr of Page Directroy Table(PDT)
	    uint32_t flags;                             // Process flag
	    char name[PROC_NAME_LEN + 1];               // Process name
	    list_entry_t list_link;                     // Process link list 
	    list_entry_t hash_link;                     // Process hash list
	};
	```
	- state：进程所处的状态
	- parent：用户进程的父进程；只有一个进程没有父进程，即内核创建的第一个内核线程idleproc
	- mm：内存管理信息，包括内存映射列表，页表指针等。mm用于虚存管理，在实际的OS中，内核线程常驻内存，不需要考虑swap page的问题。用户进程才考虑进程用户内存空间的swap page问题，这时mm才会发挥作用
	- context：进程的上下文，用于进程切换。 使用context保存寄存器使得在内核态中能够进行上下文切换
	- tf：中断帧的指针，执行内核栈的某个位置；当进程从用户空间跳到内核空间时，中断帧记录了进程在被中断前的状态。 当内核需要跳回用户空间时，需要调整中断帧以恢复让进程继续执行的各寄存器的值
	- cr3：cr3保存页表的物理地址，进程切换的时候方便直接使用lcr3实现页表切换，避免每次都根据mm来计算cr3
	- kstack：每个线程都有一个内核栈，并且位于内核地址空间的不同位置
	- current：当前占用CPU且处于“运行”状态进程控制块指针
	- initproc：本实验中，指向一个内核线程。之后，此指针指向第一个用户态进程
	- hash_list：所有进程控制块的哈希表
	- proc_list：所有进程控制块的双向线性列表


#### 创建并执行内核线程




#### 执行流程概述

* 初始化
	- kern_init函数中，当完成虚拟内存初始化工作后，就调用了proc_init函数
	- proc_init函数完成了idleproc内核线程和initproc内核线程的创建和复制工作
	- idleproc内核线程的工作就是不停地查询，看是否有其他内核线程可以执行了。 如果有，马上让调度器选择那个内核线程执行
	- 调用kernel_thread函数来创建initproc内核线程