## Prac1��make

#### 1.1 ucore.img�������?

> $ make "V="  # ��ӡmakeִ������Щ����

1. ��������kernel

* ����������������.o�ļ�
* ����.o�ļ������ɿ�ִ���ļ�kernel

```
...

+ ld bin/kernel
ld -m    elf_i386 -nostdlib -T tools/kernel.ld -o bin/kernel  obj/kern/init/init.o obj/kern/libs/readline.o obj/kern/libs/stdio.o obj/kern/debug/kdebug.o obj/kern/debug/kmonitor.o obj/kern/debug/panic.o obj/kern/driver/clock.o obj/kern/driver/console.o obj/kern/driver/intr.o obj/kern/driver/picirq.o obj/kern/trap/trap.o obj/kern/trap/trapentry.o obj/kern/trap/vectors.o obj/kern/mm/pmm.o  obj/libs/printfmt.o obj/libs/string.o
```

2. ��������bootblock
* ����������������.o�ļ�
* ����.o�ļ�������bootblock.o�ļ�
* ͨ��objcopy����bootblock.oת��Ϊbootblock.out�ļ���objcopy -S -O binary, remove all symbols and relocation information, and then generate binary file��
* ͨ��sign����bootblock.outת����bootblock�ļ�

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

3. ����ucore.img

* ʹ��dd�����/dev/zero, bootblock, kernel��������ucore.img
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


#### 1.2 Ӳ������������������

���Ͽ�֪��Ӳ�������������ļ���sign�������ɣ�

> sign bootblock.out bootblock

ͨ���Ķ�sign.cԴ�ļ���֪��

* �����ļ����ܴ���510�ֽ�
* ������������СΪ512�ֽڣ�����55AA����

```
buf[510] = 0x55;
buf[511] = 0xAA;
```

## Prac2��ʹ��qemuִ�в�����

#### 2.1 ����bootloader�ķ���

1. ���Է�ʽ����qemu

```
qemu-system-i386 -S -s -d in_asm -D bin/q.log -monitor stdio -hda bin/ucore.img -serial null
```

2. ����gdb�������öϵ㵽0x7c00��

```
gdb -q -x tools/lab1init
```

lab1init���ݣ�

```
file bin/kernel
target remote :1234
set architecture i8086
b *0x7c00
continue
x /2i $pc
```


## Prac3������bootloader���뱣��ģʽ�Ĺ���

#### 3.1 Ϊ�ο���A20���Լ���ο���A20

A20
* A20�ĳ�����Ϊ�˼������ڵ�8086 CPU��20λ���ߵ�Ѱַ��ʽ
* ����ʱ��A20��ַ�߿��������εģ�֪��ϵͳ���ͨ��һ����IO����ȥ����
* ��A20��ַ�߿��ƽ�ֹʱ�����������8086�����У�1MB���ϵĵ�ַ�ǲ��ɷ��ʵ�
* ��ʵģʽ�£�Ҫ���ʸ߶��ڴ棬������ر����

��ο���A20?

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


#### 3.2 ��γ�ʼ��GDT��

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

#### 3.3 ���ʹ�ܺͽ��뱣��ģʽ

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

## Prac4������bootloader����ELF��ʽOS�Ĺ���

#### 4.1 bootloader��ζ�ȡӲ��������

* bootloader����Ӳ�̲���LBAģʽ��PIO��Program IO����ʽ�������е�IO������ͨ��CPU����Ӳ�̵�IO��ַ�Ĵ�����ɵ�

* IO��ַ�Ͷ�Ӧ����
    * 0x1f0�������ݣ���0x1f7��Ϊæ״̬ʱ�����Զ�
    * 0x1f2��Ҫ��д��������
    * 0x1f3��LBA����0-7λ
    * 0x1f4��LBA����8-15λ
    * 0x1f5��LBA����16-23λ
    * 0x1f6��0-3λΪLBA����24-27λ����4λ��0Ϊ���̣�1Ϊ����
    * 0x1f7��״̬������Ĵ���������ʱ�ȸ�����ٶ�ȡ���������æ״̬���ʹ�0x1f0�˿ڶ�����

* ��ǰӲ�������Ǵ洢��Ӳ�������У�һ��������СΪ512�ֽڡ���һ�����������̴�������
    * �ȴ�����׼����
    * ������ȡ����������
    * �ȴ�����׼����
    * ���������ݶ���ָ���ڴ�

* boot/bootmain.c�ж�Ӧ����ʵ�֣�

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

#### 4.2 bootloader��μ���ELF��ʽ��OS��

* ���ȶ�ȡǰ8���������������ݵ�struct elfhdr�ṹ��ָ��ELFHDR�С�ͨ��ELFHDR�е�e_magic�ж϶�ȡ�����Ƿ���ELF�ļ���ʽ��
* ͨ��ELFHDR��e_phoff��program header��λ��ƫ�ƣ�����ȡ��Program header�����ڵ�ַ��ELFHDR + ELFHDR->e_phoff��������ڵ�ַת��Ϊstruct proghdr�ṹ��ָ��ph���Ӷ����Զ�ȡ��program header���е����ݣ�Ȼ�����ͨ��e_phnum�ֶεõ�program header��������Ŀ�������Ϳ���ͨ��ָ���������ζ�ȡÿ��program header������ݣ�
* ����ÿ��program header��������ݣ���ָ����ƫ�ƴ���ph->p_offset����ȡָ����С��ph->p_memsz���Ĵ������ݵ�ָ�����ڴ��ַ����ph->p_va�����Ӷ����OS����ļ��أ�
* ��ת��������ڵ�ַ����ELFHDR->e_entry��ִ�г���

* �������£�
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

## Prac5��ʵ�ֺ������ö�ջ���ٺ���

#### 5.1 ʵ�ֺ���print_stackframe

* ����ͨ��read_ebp()��read_eip()��ȡ��ǰebp��eip
* ��ǰ�����Ĳ����б�ӵ�ǰebpλ������8���ֽڴ��� (uint32_t *)ebp + 2����ʼ�洢
* �ϴκ�������λ�ã�eip���洢�ڵ�ǰebp����4���ֽڴ�����((uint32_t *)ebp)[1]��(uint32_t *)ebp + 1
* �ϴκ���ջ֡λ�ã�ebp���洢�ڵ�ǰebp��������ǰebpλ�õ�ֵ((uint32_t *)ebp)[0]��
* ���ϴε�ebp��eip��Ϊ��ǰebp��eip�����δ�ѭ��������ֱ��������ã�ebp == 0���������ջ��ȣ�i >= STACKFRAME_DEPTH��
* �������£�

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

## Prac6�������жϳ�ʼ���ʹ���

#### 6.1 �ж����������ж���������һ������ռ�����ֽڣ� �����ļ�λ�����жϴ���������ڣ�

* �ж��������������ݽṹ������ʾ��

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

* ���Ͽ�֪��һ������ռ16+16+5+3+4+1+2+1+16=64�ֽ�
* ����gd_ss�洢segment selector��gd_off_15_0��gd_off_31_16Ϊoffset��ͨ����Щ�ֶο����ҵ��жϴ�������λ�á�


#### 6.2 �����ж��������ʼ������idt_init����idt_init�����У����ζ������ж���ڽ��г�ʼ����

* ����SETGATE�����ûһ��struct gatedesc�ṹ�����ݣ�����__vectors�����д���Ŷ�Ӧ��ÿ���жϷ������̵�ƫ�Ƶ�ַ

* ��������

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

#### 6.3 �����жϴ�����trap����ʱ���жϽ��д���ʹ����ϵͳÿ����100��ʱ���жϺ󣬵���print_ticks�ӳ�������Ļ��ӡһ�����֡�

* ���е��жϵ������ջ��ߵ�trap_dispatch�����ڣ�Ȼ���ٸ����жϺŽ��зַ�
* �������£�

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


