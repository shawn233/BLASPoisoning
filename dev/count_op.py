'''
Author: shawn233
Date: 2021-03-07 10:15:54
LastEditors: shawn233
LastEditTime: 2021-03-07 10:38:46
Description: file content
'''


def main():
    opcodes = [
        "ja", "jnbe", "jae", "jnb", "jb", "jnae", "jbe", "jna", "je", "jne",
        "jg", "jnle" "jge", "jnl", "jl", "jnge", "jle", "jng",
        "jc", "jnc", "jz", "jnz", "js", "jns", "jo", "jno", "jp", "jpe", "jnp", "jpo",
        "test",
        "other"
    ]

    count_dict = {op: 0 for op in opcodes}
    
    with open("../libopenblas.a.rev", "r") as blas:
        for line in blas:
            # take the longest matching opcode
            hit_length = 0
            hit_op = "other"
            for op in opcodes:
                if op in line:
                    if len(op) > hit_length:
                        hit_length = len(op)
                        hit_op = op
            count_dict[hit_op] += 1
    
    for op in opcodes:
        print(f"{op}\t{count_dict[op]}")



if __name__ == "__main__":
    main()