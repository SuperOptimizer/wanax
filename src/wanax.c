#include "../include/wanax.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* MIPS I Emulator
 * This emulator implements the core MIPS I instruction set
 * with 32 general purpose registers and basic memory operations.
 */

#define MEMORY_SIZE (1024 * 1024)  // 1MB of memory
#define STACK_START (MEMORY_SIZE - 4)

// MIPS register file
typedef struct {
    uint32_t regs[32];
    uint32_t pc;       // Program counter
    uint32_t hi, lo;   // Multiply/divide results
} MipsRegisters;

// MIPS emulator state
typedef struct {
    MipsRegisters regs;
    uint8_t *memory;
} MipsEmulator;

// Register names for debugging
const char* reg_names[32] = {
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
    "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"
};

// Initialize the emulator
MipsEmulator* mips_init() {
    MipsEmulator *emu = (MipsEmulator*)malloc(sizeof(MipsEmulator));
    if (!emu) {
        fprintf(stderr, "Failed to allocate emulator\n");
        return NULL;
    }
    
    // Allocate memory
    emu->memory = (uint8_t*)calloc(MEMORY_SIZE, 1);
    if (!emu->memory) {
        fprintf(stderr, "Failed to allocate memory\n");
        free(emu);
        return NULL;
    }
    
    // Initialize registers
    memset(&emu->regs, 0, sizeof(MipsRegisters));
    emu->regs.regs[29] = STACK_START;  // Initialize stack pointer
    
    return emu;
}

// Free emulator resources
void mips_cleanup(MipsEmulator *emu) {
    if (emu) {
        free(emu->memory);
        free(emu);
    }
}

// Memory access functions
uint32_t mem_read_word(MipsEmulator *emu, uint32_t addr) {
    if (addr >= MEMORY_SIZE - 3) {
        fprintf(stderr, "Memory read out of bounds: 0x%08x\n", addr);
        return 0;
    }
    
    return (emu->memory[addr] << 24) |
           (emu->memory[addr + 1] << 16) |
           (emu->memory[addr + 2] << 8) |
           (emu->memory[addr + 3]);
}

void mem_write_word(MipsEmulator *emu, uint32_t addr, uint32_t value) {
    if (addr >= MEMORY_SIZE - 3) {
        fprintf(stderr, "Memory write out of bounds: 0x%08x\n", addr);
        return;
    }
    
    emu->memory[addr] = (value >> 24) & 0xFF;
    emu->memory[addr + 1] = (value >> 16) & 0xFF;
    emu->memory[addr + 2] = (value >> 8) & 0xFF;
    emu->memory[addr + 3] = value & 0xFF;
}

uint8_t mem_read_byte(MipsEmulator *emu, uint32_t addr) {
    if (addr >= MEMORY_SIZE) {
        fprintf(stderr, "Memory read out of bounds: 0x%08x\n", addr);
        return 0;
    }
    
    return emu->memory[addr];
}

void mem_write_byte(MipsEmulator *emu, uint32_t addr, uint8_t value) {
    if (addr >= MEMORY_SIZE) {
        fprintf(stderr, "Memory write out of bounds: 0x%08x\n", addr);
        return;
    }
    
    emu->memory[addr] = value;
}

// Instruction decoding
typedef enum {
    R_TYPE,
    I_TYPE,
    J_TYPE
} InstructionType;

typedef struct {
    InstructionType type;
    uint8_t opcode;
    uint8_t rs;
    uint8_t rt;
    uint8_t rd;
    uint8_t shamt;
    uint8_t funct;
    uint16_t immediate;
    uint32_t address;
} DecodedInstruction;

DecodedInstruction decode_instruction(uint32_t instruction) {
    DecodedInstruction decoded;
    
    decoded.opcode = (instruction >> 26) & 0x3F;
    
    if (decoded.opcode == 0) {
        // R-type instruction
        decoded.type = R_TYPE;
        decoded.rs = (instruction >> 21) & 0x1F;
        decoded.rt = (instruction >> 16) & 0x1F;
        decoded.rd = (instruction >> 11) & 0x1F;
        decoded.shamt = (instruction >> 6) & 0x1F;
        decoded.funct = instruction & 0x3F;
        decoded.immediate = 0;
        decoded.address = 0;
    } else if (decoded.opcode == 0x02 || decoded.opcode == 0x03) {
        // J-type instruction
        decoded.type = J_TYPE;
        decoded.rs = 0;
        decoded.rt = 0;
        decoded.rd = 0;
        decoded.shamt = 0;
        decoded.funct = 0;
        decoded.immediate = 0;
        decoded.address = instruction & 0x03FFFFFF;
    } else {
        // I-type instruction
        decoded.type = I_TYPE;
        decoded.rs = (instruction >> 21) & 0x1F;
        decoded.rt = (instruction >> 16) & 0x1F;
        decoded.rd = 0;
        decoded.shamt = 0;
        decoded.funct = 0;
        decoded.immediate = instruction & 0xFFFF;
        decoded.address = 0;
    }
    
    return decoded;
}

// Sign extension helper
uint32_t sign_extend_16(uint16_t value) {
    return (value & 0x8000) ? (0xFFFF0000 | value) : value;
}

// Execute a single instruction
void execute_instruction(MipsEmulator *emu, uint32_t instruction) {
    DecodedInstruction decoded = decode_instruction(instruction);
    uint32_t next_pc = emu->regs.pc + 4;
    uint32_t temp;
    
    // Always ensure $zero is 0
    emu->regs.regs[0] = 0;
    
    switch (decoded.type) {
        case R_TYPE:
            switch (decoded.funct) {
                case 0x00: // SLL - Shift left logical
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rt] << decoded.shamt;
                    break;
                    
                case 0x02: // SRL - Shift right logical
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rt] >> decoded.shamt;
                    break;
                    
                case 0x03: // SRA - Shift right arithmetic
                    emu->regs.regs[decoded.rd] = (int32_t)emu->regs.regs[decoded.rt] >> decoded.shamt;
                    break;
                    
                case 0x04: // SLLV - Shift left logical variable
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rt] << (emu->regs.regs[decoded.rs] & 0x1F);
                    break;
                    
                case 0x06: // SRLV - Shift right logical variable
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rt] >> (emu->regs.regs[decoded.rs] & 0x1F);
                    break;
                    
                case 0x07: // SRAV - Shift right arithmetic variable
                    emu->regs.regs[decoded.rd] = (int32_t)emu->regs.regs[decoded.rt] >> (emu->regs.regs[decoded.rs] & 0x1F);
                    break;
                    
                case 0x08: // JR - Jump register
                    next_pc = emu->regs.regs[decoded.rs];
                    break;
                    
                case 0x09: // JALR - Jump and link register
                    emu->regs.regs[decoded.rd] = emu->regs.pc + 8;
                    next_pc = emu->regs.regs[decoded.rs];
                    break;
                    
                case 0x20: // ADD - Add
                    emu->regs.regs[decoded.rd] = (int32_t)emu->regs.regs[decoded.rs] + (int32_t)emu->regs.regs[decoded.rt];
                    break;
                    
                case 0x21: // ADDU - Add unsigned
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rs] + emu->regs.regs[decoded.rt];
                    break;
                    
                case 0x22: // SUB - Subtract
                    emu->regs.regs[decoded.rd] = (int32_t)emu->regs.regs[decoded.rs] - (int32_t)emu->regs.regs[decoded.rt];
                    break;
                    
                case 0x23: // SUBU - Subtract unsigned
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rs] - emu->regs.regs[decoded.rt];
                    break;
                    
                case 0x24: // AND - Bitwise and
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rs] & emu->regs.regs[decoded.rt];
                    break;
                    
                case 0x25: // OR - Bitwise or
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rs] | emu->regs.regs[decoded.rt];
                    break;
                    
                case 0x26: // XOR - Bitwise exclusive or
                    emu->regs.regs[decoded.rd] = emu->regs.regs[decoded.rs] ^ emu->regs.regs[decoded.rt];
                    break;
                    
                case 0x27: // NOR - Bitwise nor
                    emu->regs.regs[decoded.rd] = ~(emu->regs.regs[decoded.rs] | emu->regs.regs[decoded.rt]);
                    break;
                    
                case 0x2A: // SLT - Set on less than
                    emu->regs.regs[decoded.rd] = ((int32_t)emu->regs.regs[decoded.rs] < (int32_t)emu->regs.regs[decoded.rt]) ? 1 : 0;
                    break;
                    
                case 0x2B: // SLTU - Set on less than unsigned
                    emu->regs.regs[decoded.rd] = (emu->regs.regs[decoded.rs] < emu->regs.regs[decoded.rt]) ? 1 : 0;
                    break;
                    
                case 0x18: // MULT - Multiply
                    {
                        int64_t result = (int64_t)(int32_t)emu->regs.regs[decoded.rs] * (int64_t)(int32_t)emu->regs.regs[decoded.rt];
                        emu->regs.lo = (uint32_t)result;
                        emu->regs.hi = (uint32_t)(result >> 32);
                    }
                    break;
                    
                case 0x19: // MULTU - Multiply unsigned
                    {
                        uint64_t result = (uint64_t)emu->regs.regs[decoded.rs] * (uint64_t)emu->regs.regs[decoded.rt];
                        emu->regs.lo = (uint32_t)result;
                        emu->regs.hi = (uint32_t)(result >> 32);
                    }
                    break;
                    
                case 0x1A: // DIV - Divide
                    if (emu->regs.regs[decoded.rt] != 0) {
                        emu->regs.lo = (int32_t)emu->regs.regs[decoded.rs] / (int32_t)emu->regs.regs[decoded.rt];
                        emu->regs.hi = (int32_t)emu->regs.regs[decoded.rs] % (int32_t)emu->regs.regs[decoded.rt];
                    }
                    break;
                    
                case 0x1B: // DIVU - Divide unsigned
                    if (emu->regs.regs[decoded.rt] != 0) {
                        emu->regs.lo = emu->regs.regs[decoded.rs] / emu->regs.regs[decoded.rt];
                        emu->regs.hi = emu->regs.regs[decoded.rs] % emu->regs.regs[decoded.rt];
                    }
                    break;
                    
                case 0x10: // MFHI - Move from HI
                    emu->regs.regs[decoded.rd] = emu->regs.hi;
                    break;
                    
                case 0x12: // MFLO - Move from LO
                    emu->regs.regs[decoded.rd] = emu->regs.lo;
                    break;
                    
                case 0x11: // MTHI - Move to HI
                    emu->regs.hi = emu->regs.regs[decoded.rs];
                    break;
                    
                case 0x13: // MTLO - Move to LO
                    emu->regs.lo = emu->regs.regs[decoded.rs];
                    break;
                    
                default:
                    fprintf(stderr, "Unimplemented R-type instruction: function 0x%02X\n", decoded.funct);
                    break;
            }
            break;
            
        case I_TYPE:
            switch (decoded.opcode) {
                case 0x08: // ADDI - Add immediate
                    emu->regs.regs[decoded.rt] = (int32_t)emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    break;
                    
                case 0x09: // ADDIU - Add immediate unsigned
                    emu->regs.regs[decoded.rt] = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    break;
                    
                case 0x0C: // ANDI - And immediate
                    emu->regs.regs[decoded.rt] = emu->regs.regs[decoded.rs] & decoded.immediate;
                    break;
                    
                case 0x0D: // ORI - Or immediate
                    emu->regs.regs[decoded.rt] = emu->regs.regs[decoded.rs] | decoded.immediate;
                    break;
                    
                case 0x0E: // XORI - Exclusive or immediate
                    emu->regs.regs[decoded.rt] = emu->regs.regs[decoded.rs] ^ decoded.immediate;
                    break;
                    
                case 0x0A: // SLTI - Set on less than immediate
                    emu->regs.regs[decoded.rt] = ((int32_t)emu->regs.regs[decoded.rs] < (int32_t)sign_extend_16(decoded.immediate)) ? 1 : 0;
                    break;
                    
                case 0x0B: // SLTIU - Set on less than immediate unsigned
                    emu->regs.regs[decoded.rt] = (emu->regs.regs[decoded.rs] < sign_extend_16(decoded.immediate)) ? 1 : 0;
                    break;
                    
                case 0x04: // BEQ - Branch on equal
                    if (emu->regs.regs[decoded.rs] == emu->regs.regs[decoded.rt]) {
                        next_pc = emu->regs.pc + 4 + (sign_extend_16(decoded.immediate) << 2);
                    }
                    break;
                    
                case 0x05: // BNE - Branch on not equal
                    if (emu->regs.regs[decoded.rs] != emu->regs.regs[decoded.rt]) {
                        next_pc = emu->regs.pc + 4 + (sign_extend_16(decoded.immediate) << 2);
                    }
                    break;
                    
                case 0x06: // BLEZ - Branch on less than or equal to zero
                    if ((int32_t)emu->regs.regs[decoded.rs] <= 0) {
                        next_pc = emu->regs.pc + 4 + (sign_extend_16(decoded.immediate) << 2);
                    }
                    break;
                    
                case 0x07: // BGTZ - Branch on greater than zero
                    if ((int32_t)emu->regs.regs[decoded.rs] > 0) {
                        next_pc = emu->regs.pc + 4 + (sign_extend_16(decoded.immediate) << 2);
                    }
                    break;
                    
                case 0x01: // BLTZ/BGEZ - Branch on less than zero/Branch on greater than or equal to zero
                    if (decoded.rt == 0) { // BLTZ
                        if ((int32_t)emu->regs.regs[decoded.rs] < 0) {
                            next_pc = emu->regs.pc + 4 + (sign_extend_16(decoded.immediate) << 2);
                        }
                    } else if (decoded.rt == 1) { // BGEZ
                        if ((int32_t)emu->regs.regs[decoded.rs] >= 0) {
                            next_pc = emu->regs.pc + 4 + (sign_extend_16(decoded.immediate) << 2);
                        }
                    }
                    break;
                    
                case 0x23: // LW - Load word
                    temp = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    emu->regs.regs[decoded.rt] = mem_read_word(emu, temp);
                    break;
                    
                case 0x21: // LH - Load halfword
                    temp = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    emu->regs.regs[decoded.rt] = sign_extend_16(mem_read_word(emu, temp) & 0xFFFF);
                    break;
                    
                case 0x25: // LHU - Load halfword unsigned
                    temp = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    emu->regs.regs[decoded.rt] = mem_read_word(emu, temp) & 0xFFFF;
                    break;
                    
                case 0x20: // LB - Load byte
                    temp = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    emu->regs.regs[decoded.rt] = (int8_t)mem_read_byte(emu, temp);
                    break;
                    
                case 0x24: // LBU - Load byte unsigned
                    temp = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    emu->regs.regs[decoded.rt] = mem_read_byte(emu, temp);
                    break;
                    
                case 0x2B: // SW - Store word
                    temp = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    mem_write_word(emu, temp, emu->regs.regs[decoded.rt]);
                    break;
                    
                case 0x29: // SH - Store halfword
                    temp = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    mem_write_word(emu, temp, emu->regs.regs[decoded.rt] & 0xFFFF);
                    break;
                    
                case 0x28: // SB - Store byte
                    temp = emu->regs.regs[decoded.rs] + sign_extend_16(decoded.immediate);
                    mem_write_byte(emu, temp, emu->regs.regs[decoded.rt] & 0xFF);
                    break;
                    
                case 0x0F: // LUI - Load upper immediate
                    emu->regs.regs[decoded.rt] = decoded.immediate << 16;
                    break;
                    
                default:
                    fprintf(stderr, "Unimplemented I-type instruction: opcode 0x%02X\n", decoded.opcode);
                    break;
            }
            break;
            
        case J_TYPE:
            switch (decoded.opcode) {
                case 0x02: // J - Jump
                    next_pc = (emu->regs.pc & 0xF0000000) | (decoded.address << 2);
                    break;
                    
                case 0x03: // JAL - Jump and link
                    emu->regs.regs[31] = emu->regs.pc + 8;
                    next_pc = (emu->regs.pc & 0xF0000000) | (decoded.address << 2);
                    break;
                    
                default:
                    fprintf(stderr, "Unimplemented J-type instruction: opcode 0x%02X\n", decoded.opcode);
                    break;
            }
            break;
    }
    
    // Update PC
    emu->regs.pc = next_pc;
}

// Load a program into memory
int load_program(MipsEmulator *emu, const char *filename, uint32_t load_addr) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return 0;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (load_addr + size > MEMORY_SIZE) {
        fprintf(stderr, "Program too large to fit in memory\n");
        fclose(f);
        return 0;
    }
    
    size_t read = fread(emu->memory + load_addr, 1, size, f);
    fclose(f);
    
    if (read != size) {
        fprintf(stderr, "Failed to read entire program\n");
        return 0;
    }
    
    return 1;
}

// Run the emulator
void run_emulator(MipsEmulator *emu, uint32_t start_addr, uint32_t max_instructions) {
    emu->regs.pc = start_addr;
    uint32_t instruction_count = 0;
    
    while (instruction_count < max_instructions) {
        uint32_t instruction = mem_read_word(emu, emu->regs.pc);
        execute_instruction(emu, instruction);
        instruction_count++;
        
        // Check for termination (e.g., if PC is 0)
        if (emu->regs.pc == 0) {
            printf("Program terminated (PC = 0)\n");
            break;
        }
    }
    
    if (instruction_count >= max_instructions) {
        printf("Reached maximum instruction count (%u)\n", max_instructions);
    }
}

// Debug: Dump registers
void dump_registers(MipsEmulator *emu) {
    printf("Registers:\n");
    for (int i = 0; i < 32; i += 4) {
        printf("%s: 0x%08X  %s: 0x%08X  %s: 0x%08X  %s: 0x%08X\n",
               reg_names[i], emu->regs.regs[i],
               reg_names[i+1], emu->regs.regs[i+1],
               reg_names[i+2], emu->regs.regs[i+2],
               reg_names[i+3], emu->regs.regs[i+3]);
    }
    printf("PC: 0x%08X  HI: 0x%08X  LO: 0x%08X\n", 
           emu->regs.pc, emu->regs.hi, emu->regs.lo);
}

// Example usage
int main() {
    MipsEmulator *emu = mips_init();
    if (!emu) {
        return 1;
    }
    
    // Example: Load a program and run it
    // Uncomment and modify to use with actual MIPS binary files
    /*
    if (!load_program(emu, "program.bin", 0x1000)) {
        mips_cleanup(emu);
        return 1;
    }
    
    run_emulator(emu, 0x1000, 1000000);
    dump_registers(emu);
    */
    
    // For now, just demonstrate a simple program directly in memory
    // This program calculates factorial of 5
    
    // li $a0, 5       # Load 5 into $a0
    mem_write_word(emu, 0x1000, 0x20040005);
    
    // jal factorial   # Call factorial function
    mem_write_word(emu, 0x1004, 0x0C000402);
    
    // break           # End program
    mem_write_word(emu, 0x1008, 0x0000000D);
    
    // factorial:      # Function to calculate factorial
    // addi $sp, $sp, -8  # Allocate stack space
    mem_write_word(emu, 0x1008, 0x23BDFFF8);
    
    // sw $ra, 4($sp)  # Save return address
    mem_write_word(emu, 0x100C, 0xAFBF0004);
    
    // sw $a0, 0($sp)  # Save argument
    mem_write_word(emu, 0x1010, 0xAFA40000);
    
    // slti $t0, $a0, 2  # Check if n < 2
    mem_write_word(emu, 0x1014, 0x28880002);
    
    // beq $t0, $zero, recurse  # If n >= 2, recurse
    mem_write_word(emu, 0x1018, 0x11000003);
    
    // li $v0, 1       # Base case: return 1
    mem_write_word(emu, 0x101C, 0x24020001);
    
    // j factorial_return
    mem_write_word(emu, 0x1020, 0x0800040D);
    
    // recurse:
    // addi $a0, $a0, -1  # n-1
    mem_write_word(emu, 0x1024, 0x2084FFFF);
    
    // jal factorial   # Call factorial(n-1)
    mem_write_word(emu, 0x1028, 0x0C000402);
    
    // lw $a0, 0($sp)  # Restore n
    mem_write_word(emu, 0x102C, 0x8FA40000);
    
    // mul $v0, $a0, $v0  # n * factorial(n-1)
    mem_write_word(emu, 0x1030, 0x00820018);
    
    // factorial_return:
    // lw $ra, 4($sp)  # Restore return address
    mem_write_word(emu, 0x1034, 0x8FBF0004);
    
    // addi $sp, $sp, 8  # Restore stack pointer
    mem_write_word(emu, 0x1038, 0x23BD0008);
    
    // jr $ra          # Return
    mem_write_word(emu, 0x103C, 0x03E00008);
    
    // Run the program
    run_emulator(emu, 0x1000, 1000);
    
    // Display results
    printf("Factorial of 5 = %u\n", emu->regs.regs[2]);  // Result in $v0
    dump_registers(emu);




    //explain mips_cleanup AI!

    mips_cleanup(emu);
    return 0;
}
