## Prac1：make

#### 1.1 ucore.img如何生成?

> $ make "V="  # 打印make执行了哪些命令

1. 编译生成kernel

* 编译生成所依赖的.o文件
* 链接.o文件，生成可执行文件kernel

```
...

+ ld bin/kernel
ld -m    elf_i386 -nostdlib -T tools/kernel.ld -o bin/kernel  obj/kern/init/init.o obj/kern/libs/readline.o obj/kern/libs/stdio.o obj/kern/debug/kdebug.o obj/kern/debug/kmonitor.o obj/kern/debug/panic.o obj/kern/driver/clock.o obj/kern/driver/console.o obj/kern/driver/intr.o obj/kern/driver/picirq.o obj/kern/trap/trap.o obj/kern/trap/trapentry.o obj/kern/trap/vectors.o obj/kern/mm/pmm.o  obj/libs/printfmt.o obj/libs/string.o
```

2. 编译生成bootblock
* 编译生成所依赖的.o文件
* 链接.o文件，生成bootblock.o文件
* 通过objcopy，将bootblock.o转换为bootblock.out文件（objcopy -S -O binary, remove all symbols and relocation information, and then generate binary file）
* 通过sign，将bootblock.out转换成bootblock文件

```
+ cc boot/bootasm.S
gcc -Iboot/ -fno-builtin -Wall -ggdb -m32 -gstabs -nostdinc  -fno-stack-protector -Ilibs/ -Os -nostdinc -c boot/bootasm.S -o obj/boot/bootasm.o
+ cc boot/bootmain.c
gcc -Iboot/ -fno-builtin -Wall -ggdb -m32 -gstabs -nostdinc  -fno-stack-protector -Ilibs/ -Os -nostdinc -c boot/bootmain.c -o obj/boot/bootmain.o
+ cc tools/sign.c
gcc -Itools/ -g -Wall -O2 -c tools/sign.c -o obj/sign/tools/sign.o
gcc -g -Wall -O2 obj/sign/tools/sign.o -o bin/sign
+ ld bin/bootblock
ld -m    elf_i386 -nostdlib -N -e start -Ttext 0x7C00 obj/boot/bootasm.o obj/boot/bootmain.o -o obj/bootblock.o
objdump -S obj/bootblock.o > obj/bootblock.asm
objcopy -S -O binary obj/bootblock.o obj/bootblock.out
bin/sign obj/bootblock.out bin/bootblock
'obj/bootblock.out' size: 472 bytes
build 512 bytes boot sector: 'bin/bootblock' success!
```

3. 生成ucore.img

* 使用dd命令，将/dev/zero, bootblock, kernel拷贝生成ucore.img
* [dd](http://www.cnblogs.com/dkblog/archive/2009/09/18/1980715.html)

```
dd if=/dev/zero of=bin/ucore.img count=10000
10000+0 records in
10000+0 records out
5120000 bytes (5.1 MB) copied, 0.0132338 s, 387 MB/s
dd if=bin/bootblock of=bin/ucore.img conv=notrunc
1+0 records in
1+0 records out
512 bytes (512 B) copied, 0.00016476 s, 3.1 MB/s
dd if=bin/kernel of=bin/ucore.img seek=1 conv=notrunc
138+1 records in
138+1 records out
70783 bytes (71 kB) copied, 0.000345263 s, 205 MB/s
```


#### 1.2 硬盘主引导扇区的特征

由上可知，硬盘主引导扇区文件由sign工具生成：

> sign bootblock.out bootblock

通过阅读sign.c源文件可知：

* 输入文件不能大于510字节
* 主引导扇区大小为512字节，且以55AA结束

```
buf[510] = 0x55;
buf[511] = 0xAA;
```

## Prac2：使用qemu执行并调试

#### 2.1 调试bootloader的方法

1. 调试方式启动qemu

```
qemu-system-i386 -S -s -d in_asm -D bin/q.log -monitor stdio -hda bin/ucore.img -serial null
```

2. 启动gdb，并设置断点到0x7c00处

```
gdb -q -x tools/lab1init
```

lab1init内容：

```
file bin/kernel
target remote :1234
set architecture i8086
b *0x7c00
continue
x /2i $pc
```


## Prac3：分析bootloader进入保护模式的过程

#### 3.1 为何开启A20，以及如何开启A20

A20
* A20的出现是为了兼容早期的8086 CPU的20位总线的寻址方式
* 启动时，A20地址线控制是屏蔽的，知道系统软件通过一定的IO操作去打开它
* 当A20地址线控制禁止时，程序就像在8086中运行，1MB以上的地址是不可访问的
* 在实模式下，要访问高端内存，这个开关必须打开

如何开启A20?

```
    # Enable A20:
    #  For backwards compatibility with the earliest PCs, physical
    #  address line 20 is tied low, so that addresses higher than
    #  1MB wrap around to zero by default. This code undoes this.
seta20.1:
    inb $0x64, %al                                  # Wait for not busy(8042 input buffer empty).
    testb $0x2, %al
    jnz seta20.1

    movb $0xd1, %al                                 # 0xd1 -> port 0x64
    outb %al, $0x64                                 # 0xd1 means: write data to 8042's P2 port

seta20.2:
    inb $0x64, %al                                  # Wait for not busy(8042 input buffer empty).
    testb $0x2, %al
    jnz seta20.2

    movb $0xdf, %al                                 # 0xdf -> port 0x60
    outb %al, $0x60                                 # 0xdf = 11011111, means set P2's A20 bit(the 1 bit) to 1
```


#### 3.2 如何初始化GDT表

```
    # Switch from real to protected mode, using a bootstrap GDT
    # and segment translation that makes virtual addresses
    # identical to physical addresses, so that the
    # effective memory map does not change during the switch.
    lgdt gdtdesc
    movl %cr0, %eax
    orl $CR0_PE_ON, %eax
    movl %eax, %cr0
    
...

# Bootstrap GDT
.p2align 2                                          # force 4 byte alignment
gdt:
    SEG_NULLASM                                     # null seg
    SEG_ASM(STA_X|STA_R, 0x0, 0xffffffff)           # code seg for bootloader and kernel
    SEG_ASM(STA_W, 0x0, 0xffffffff)                 # data seg for bootloader and kernel

gdtdesc:
    .word 0x17                                      # sizeof(gdt) - 1
    .long gdt                                       # address gdt
```

#### 3.3 如何使能和进入保护模式

```
# Jump to next instruction, but in 32-bit code segment.
    # Switches processor into 32-bit mode.
    ljmp $PROT_MODE_CSEG, $protcseg

.code32                                             # Assemble for 32-bit mode
protcseg:
    # Set up the protected-mode data segment registers
    movw $PROT_MODE_DSEG, %ax                       # Our data segment selector
    movw %ax, %ds                                   # -> DS: Data Segment
    movw %ax, %es                                   # -> ES: Extra Segment
    movw %ax, %fs                                   # -> FS
    movw %ax, %gs                                   # -> GS
    movw %ax, %ss                                   # -> SS: Stack Segment

    # Set up the stack pointer and call into C. The stack region is from 0--start(0x7c00)
    movl $0x0, %ebp
    movl $start, %esp
    call bootmain
```

## Prac4：分析bootloader加载ELF格式OS的过程

#### 4.1 bootloader如何读取硬盘扇区？

* bootloader访问硬盘采用LBA模式的PIO（Program IO）方式，即所有的IO操作是通过CPU访问硬盘的IO地址寄存器完成的

* IO地址和对应功能
    * 0x1f0：读数据，当0x1f7不为忙状态时，可以读
    * 0x1f2：要读写的扇区数
    * 0x1f3：LBA参数0-7位
    * 0x1f4：LBA参数8-15位
    * 0x1f5：LBA参数16-23位
    * 0x1f6：0-3位为LBA参数24-27位，第4位：0为主盘，1为从盘
    * 0x1f7：状态和命令寄存器，操作时先给命令，再读取；如果不是忙状态，就从0x1f0端口读数据

* 当前硬盘数据是存储到硬盘扇区中，一个扇区大小为512字节。读一个扇区的流程大致如下
    * 等待磁盘准备好
    * 发出读取扇区的命令
    * 等待磁盘准备好
    * 把扇区数据读到指定内存

* boot/bootmain.c中对应代码实现：

```
/* waitdisk - wait for disk ready */
static void
waitdisk(void) {
    while ((inb(0x1F7) & 0xC0) != 0x40)
        /* do nothing */;
}

/* readsect - read a single sector at @secno into @dst */
static void
readsect(void *dst, uint32_t secno) {
    // wait for disk to be ready
    waitdisk();

    outb(0x1F2, 1);                         // count = 1
    outb(0x1F3, secno & 0xFF);
    outb(0x1F4, (secno >> 8) & 0xFF);
    outb(0x1F5, (secno >> 16) & 0xFF);
    outb(0x1F6, ((secno >> 24) & 0xF) | 0xE0);
    outb(0x1F7, 0x20);                      // cmd 0x20 - read sectors

    // wait for disk to be ready
    waitdisk();

    // read a sector
    insl(0x1F0, dst, SECTSIZE / 4);
}
```

#### 4.2 bootloader如何加载ELF格式的OS？

* 首先读取前8个磁盘扇区的内容到struct elfhdr结构的指针ELFHDR中。通过ELFHDR中的e_magic判断读取内容是否是ELF文件格式；
* 通过ELFHDR的e_phoff（program header表位置偏移），获取到Program header表的入口地址（ELFHDR + ELFHDR->e_phoff）。将入口地址转换为struct proghdr结构的指针ph，从而可以读取到program header表中的内容；然后可以通过e_phnum字段得到program header表的入口数目，这样就可以通过指针运算依次读取每个program header表的内容；
* 根据每个program header表项的内容，从指定的偏移处（ph->p_offset）读取指定大小（ph->p_memsz）的磁盘内容到指定的内存地址处（ph->p_va）；从而完成OS程序的加载；
* 跳转到程序入口地址处（ELFHDR->e_entry）执行程序；

* 代码如下：
```
void bootmain(void) {
    // read the 1st page off disk
    readseg((uintptr_t)ELFHDR, SECTSIZE * 8, 0); 

    // is this a valid ELF?
    if (ELFHDR->e_magic != ELF_MAGIC) {
        goto bad;
    }   

    struct proghdr *ph, *eph;

    // load each program segment (ignores ph flags)
    ph = (struct proghdr *)((uintptr_t)ELFHDR + ELFHDR->e_phoff);
    eph = ph + ELFHDR->e_phnum;
    for (; ph < eph; ph ++) {
        readseg(ph->p_va & 0xFFFFFF, ph->p_memsz, ph->p_offset);
    }   

    // call the entry point from the ELF header
    // note: does not return
    ((void (*)(void))(ELFHDR->e_entry & 0xFFFFFF))();

bad:
    outw(0x8A00, 0x8A00);
    outw(0x8A00, 0x8E00);

    /* do nothing */
    while (1);
}
```

## Prac5：实现函数调用堆栈跟踪函数

#### 5.1 实现函数print_stackframe

* 首先通过read_ebp()和read_eip()获取当前ebp和eip
* 当前函数的参数列表从当前ebp位置向上8个字节处（ (uint32_t *)ebp + 2）开始存储
* 上次函数调用位置（eip）存储在当前ebp向上4个字节处，即((uint32_t *)ebp)[1]或(uint32_t *)ebp + 1
* 上次函数栈帧位置（ebp）存储在当前ebp处，即当前ebp位置的值((uint32_t *)ebp)[0]。
* 将上次的ebp和eip作为当前ebp和eip，依次此循环操作，直到最初调用（ebp == 0）或者最大栈深度（i >= STACKFRAME_DEPTH）
* 代码如下：

```
void print_stackframe(void) {  

    uint32_t ebp = read_ebp(), eip = read_eip();

    int i, j;
    for (i = 0; ebp != 0 && i < STACKFRAME_DEPTH; i ++) {
        cprintf("ebp:0x%08x eip:0x%08x args:", ebp, eip);
        uint32_t *args = (uint32_t *)ebp + 2;
        for (j = 0; j < 4; j ++) {
            cprintf("0x%08x ", args[j]);
        }   
        cprintf("\n");
        print_debuginfo(eip - 1); 
        eip = ((uint32_t *)ebp)[1];
        ebp = ((uint32_t *)ebp)[0];
    }
}

```

## Prac6：完善中断初始化和处理

#### 6.1 中断描述符表（中断向量表）中一个表项占多少字节？ 其中哪几位代表中断处理代码的入口？

* 中断描述符表项数据结构如下所示：

```
/* Gate descriptors for interrupts and traps */
struct gatedesc {
    unsigned gd_off_15_0 : 16;        // low 16 bits of offset in segment
    unsigned gd_ss : 16;            // segment selector
    unsigned gd_args : 5;            // # args, 0 for interrupt/trap gates
    unsigned gd_rsv1 : 3;            // reserved(should be zero I guess)
    unsigned gd_type : 4;            // type(STS_{TG,IG32,TG32})
    unsigned gd_s : 1;                // must be 0 (system)
    unsigned gd_dpl : 2;            // descriptor(meaning new) privilege level
    unsigned gd_p : 1;                // Present
    unsigned gd_off_31_16 : 16;        // high bits of offset in segment
};
```

* 由上可知，一个表项占16+16+5+3+4+1+2+1+16=64字节
* 其中gd_ss存储segment selector，gd_off_15_0和gd_off_31_16为offset，通过这些字段可以找到中断处理的入口位置。


#### 6.2 完善中断向量表初始化函数idt_init。在idt_init函数中，依次对所有中断入口进行初始化。

* 利用SETGATE宏填充没一条struct gatedesc结构的内容，其中__vectors数组中存放着对应的每个中断服务例程的偏移地址

* 代码如下

```

#define SETGATE(gate, istrap, sel, off, dpl) {            \
    (gate).gd_off_15_0 = (uint32_t)(off) & 0xffff;        \
    (gate).gd_ss = (sel);                                \
    (gate).gd_args = 0;                                    \
    (gate).gd_rsv1 = 0;                                    \
    (gate).gd_type = (istrap) ? STS_TG32 : STS_IG32;    \
    (gate).gd_s = 0;                                    \
    (gate).gd_dpl = (dpl);                                \
    (gate).gd_p = 1;                                    \
    (gate).gd_off_31_16 = (uint32_t)(off) >> 16;        \
}

static struct gatedesc idt[256] = {{0}};

/* idt_init - initialize IDT to each of the entry points in kern/trap/vectors.S */
void idt_init(void) {
    extern uintptr_t __vectors[];
    int i;
    for (i = 0; i < sizeof(idt) / sizeof(struct gatedesc); i ++) {
        SETGATE(idt[i], 0, GD_KTEXT, __vectors[i], DPL_KERNEL);
    }   
        // set for switch from user to kernel
    SETGATE(idt[T_SWITCH_TOK], 0, GD_KTEXT, __vectors[T_SWITCH_TOK], DPL_USER);
        // load the IDT
    lidt(&idt_pd);
}

```

#### 6.3 完善中断处理函数trap，对时钟中断进行处理，使操作系统每遇到100次时钟中断后，调用print_ticks子程序，向屏幕打印一行文字。

* 所有的中断调用最终会走到trap_dispatch函数内，然后再根据中断号进行分发
* 代码如下：

```

/* trap_dispatch - dispatch based on what type of trap occurred */
static void
trap_dispatch(struct trapframe *tf) {
    char c;

    switch (tf->tf_trapno) {
    case IRQ_OFFSET + IRQ_TIMER:
        ++ticks;
        if (ticks % TICK_NUM == 0) {
            print_ticks();
        }   
        break;
    case IRQ_OFFSET + IRQ_COM1:
        c = cons_getc();
        cprintf("serial [%03d] %c\n", c, c); 
        break;
    case IRQ_OFFSET + IRQ_KBD:
        c = cons_getc();
        cprintf("kbd [%03d] %c\n", c, c); 
        break;
    //LAB1 CHALLENGE 1 : YOUR CODE you should modify below codes.
    case T_SWITCH_TOU:
    case T_SWITCH_TOK:
        panic("T_SWITCH_** ??\n");
        break;
    case IRQ_OFFSET + IRQ_IDE1:
    case IRQ_OFFSET + IRQ_IDE2:
        /* do nothing */
        break;
    default:
        // in kernel, it must be a mistake
        if ((tf->tf_cs & 3) == 0) {
            print_trapframe(tf);
            panic("unexpected trap in kernel.\n");
        }   
    }   
}
```


