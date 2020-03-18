# -*- coding: utf-8 -*-
"""
generator는 반복자와 같은 루프의 작용을 컨트롤하기 위해 쓰여지는 특별한 함수 또는 루틴.
배열이나 리스트를 리턴하는 함수와 비슷하고, 호출 할 수 있는 파라미터를 가지고 있고, 연소적인 값들을 만들어 낸다.
한번에 모든 값을 포함한 배열을 만들어서 리턴하는 대신에 yield 구문을 이용해 한 번 호출될 때마다 하나의 값만을 리턴하고, 이런 이유로 일반 반복자에 비해
아주 작은 메모리를 필요로 한다.

일반함수가 호출되면 코드의 첫 번째행 부터 시작하여 리턴(return) 구문이나, 예외(exception) 또는 (리턴을 하지않는 함수이면) 마지막 구문을
만날때까지 실행된 후, 호출자(caller)에게 모든 컨트롤을 리턴합니다. 그리고 함수가 가지고 있던 모든 내부 함수나 모든 로컬 변수는 메모리상에서 사라집니다.
같은 함수가 다시 호출되면 모든 것은 처음부터 다시 새롭게 시작됩니다.

그런데 어느날 부터 프로그래머들은 한번에 일을 다하고 영원히 사라져버리는 함수가 아닌 하나의 일을 마치면 자기가 했던 일을 기억하면서 대기하고 있다가
다시 호출되면 전의 일을 계속 이어서 하는 똑똑한 함수를 필요로 하기 시작했습니다. 그래서 만들어진 것이 제너레이터입니다.
"""


def square_numbers(nums):
    # 일반적인 반복문 형태.
    result = []
    for i in nums:
        result.append(i * i)
    return result


my_nums = square_numbers([1, 2, 3, 4, 5])
print(my_nums)


def generator_numbers(nums):
    for i in nums:
        yield i * i

    # return 되는 값은 <generator object generator_numbers at 0x7f7117012910>
    # generator는 자신이 리턴할 모든 값을 메모리에 저장하지 않기 때문에 조금 전 일반함수의 결과와 같이 한번에 리스트로 보이지 않음.
    # generator는 한 번 호출될때마다 하나의 값만을 전달 - yield 합니다.
    # 즉, yield i * i 까지는 아무런 계산을 하지 않고 다음 값에 대해서 물어보기를 기다리는 중임.


my_nums = generator_numbers([1, 2, 3, 4, 5])
print(my_nums)
print(next(my_nums))

for num in my_nums:
    print(num)