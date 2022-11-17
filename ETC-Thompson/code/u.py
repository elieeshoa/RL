# for strings s and t check for every i if t[i] is the next character to s[i]
# next character means that the next character in the alphabet, and the next
# character to z is a

def next_char(s, t):
    for i in range(len(s)):
        if s[i] == 'z':
            if t[i] != 'a':
                return False
        elif t[i] != chr(ord(s[i]) + 1):
            return False
    return True

print(next_char('zba', 'acb')) # True

# A strawberry packager starts with a pile containing `initialAmount` pounds of strawberries, and produces packages of strawberries:

# The packager looks at the first pile. If the pile does not exceed `packageMaxAmount`, the packager puts the pile into a new package.
# Otherwise, the packager splits the pile. If its weight is an even number, it's split into two equal piles. 
# If odd, it's split into two piles that differ by one pound.
# The packager puts the two new piles at the end of the line for processing. The smaller pile goes first.
# They repeat these steps until there are no remaining piles.
# Given an `initialAmount` and `packageMaxAmount`, your task is to return the number of packages of strawberries the strawberry packager will produce.

# Example. for `initialAmount = 14` and `packageMaxAmount = 3`, the output should be solution(initialAmount, packageMaxAmount) = 6.

def solution(initialAmount, packageMaxAmount):
    if initialAmount <= packageMaxAmount:
        return 1
    if initialAmount % 2 == 0:
        return solution(initialAmount // 2, packageMaxAmount) + solution(initialAmount // 2, packageMaxAmount)
    else:
        return solution(initialAmount // 2, packageMaxAmount) + solution(initialAmount // 2 + 1, packageMaxAmount)


# Your task is to implement a simplified inventory tracker system for a large retail store. 
# You are given a price list that describes the current market price for each item, and a 
# transaction log where each record contains information about one of three types of transactions - sell, discount_start, or discount_end. 
# The tracker system should process all transactions and return the total revenue from all sales.

# Each element in the price list follow this format: "<item name>: <price>", which 
# represents that item <item name> can be sold at the price of <price>.

# Transactions are provided in the following format:

# "sell <item_name>, <count>" - the store sells <count> units of <item_name>.
# "discount_start <item_name>, <discount_amount>, <max_count>" - the store announces a discount, 
# and now sells <item_name> at the price of price - <discount_amount>. However, only <max_count> 
# units can be sold at the discounted price, and all additional units are sold at the original price. 
# Note that <max count> can only be 100 at most. Also, there can only be one discount for each item at any time.
# "discount_end <item_name>" - the store announces the end of the discount for <item_name>. 
# It is guaranteed that the item <item name> has a discount at this point.

# Example For pricelist = ["item1: 100", "item2: 200"] and
# logs = [
#   "sell item1, 1",
#   "sell item1, 2",
#   "sell item2, 2",
#   "discount_start item2, 40, 1",
#   "sell item2, 2",
#   "sell item1, 1",
#   "discount_end item2",
#   "sell item2, 1"
# ]
# the output should be solution(pricelist, logs) = 1360.


def solution(pricelist, logs):
    price = {}
    for i in pricelist:
        price[i.split(':')[0]] = int(i.split(':')[1])
    revenue = 0
    discount = {}
    for i in logs:
        if i.startswith('sell'):
            item = i.split()[1][:-1]
            count = int(i.split()[2])
            if item in discount:
                if discount[item][1] >= count:
                    revenue += count * (price[item] - discount[item][0])
                    discount[item][1] -= count
                else:
                    revenue += discount[item][1] * (price[item] - discount[item][0])
                    revenue += (count - discount[item][1]) * price[item]
                    discount[item][1] = 0
            else:
                revenue += count * price[item]
        elif i.startswith('discount_start'):
            item = i.split()[1][:-1]
            discount_amount = int(i.split()[2][:-1])
            max_count = int(i.split()[3])
            discount[item] = [discount_amount, max_count]
        else:
            item = i.split()[1]
            discount.pop(item)
    return revenue
        

# test the example
pricelist = ["item1: 100", "item2: 200"]
logs = [
    "sell item1, 1",
    "sell item1, 2",
    "sell item2, 2",
    "discount_start item2, 40, 1",
    "sell item2, 2",
    "sell item1, 1",
    "discount_end item2",
    "sell item2, 1"
]
print(solution(pricelist, logs))


# A cyclic shift is the operation of rearranging the digits in a number (in decimal format) 
# by moving some digits at the end of the number to before the beginning of the number, while 
# shifting all other digits to the next position. Given two integers of the same length a and 
# b, a would be a cyclic pair of b if it's possible for a to become equal to b after performing 
# cyclic shifts on a - moving 0 or more ending digits to the beginning while shifting all other 
# digits to the next position in the same order.

# Given an array of positive integers a, your task is to count the number of cyclic pairs i and j 
# (where 0 ≤ i < j < a.length), such that a[i] and a[j] have the same number of digits and a[i] 
# is equal to a cyclic shift of a[j].

# Example

# For a = [13, 5604, 31, 2, 13, 4560, 546, 654, 456], the output should be solution(a) = 5.

# There are 5 cyclic pairs of numbers - pairs which are equal to each other after cyclic shifts.

# a[0] = 13 and a[2] = 31 (i = 0 and j = 2),
# a[0] = 13 and a[4] = 13 (i = 0 and j = 4),
# a[2] = 31 and a[4] = 13 (i = 2 and j = 4),
# a[1] = 5604 and a[5] = 4560 (i = 1 and j = 5),
# a[6] = 546 and a[7] = 654 (i = 6 and j = 7)
# Note that a[6] = 546 and a[8] = 456 are not cyclic pairs — 546 can only be paired with cyclic shift of 546, 465 and 654.
# Also, note that a[5] = 4560 and a[8] = 456 are not cyclic pairs because they have different number of digits.

def solution(a):
    count = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(str(a[i])) == len(str(a[j])):
                if str(a[i]) in str(a[j]) * 2:
                    count += 1
    return count

# faster solution using set
def solution(a):
    count = 0

    def smallest(x):
        s = str(x)
        return min([s[i:] + s[:i] for i in range(len(s))])
    for x in set(a):
        print(smallest(x))


# infinity
inf = float('inf')

# test the example
a = [13, 5604, 31, 2, 13, 4560, 546, 654, 456]
print(solution(a))