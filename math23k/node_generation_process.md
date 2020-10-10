## Graph2Tree中节点的生成过程

#### 前序表达式

```bash
[0, 3, 1, 9,  8,  1, 5, 7,  1, 5, 10]
[*, /, -, N2, N1, -, 1, N0, -, 1, N3]
```

![exp_tree](material/exp_tree.png)

![sub_tree_emb](material/sub_tree_emb.png)

#### sub-tree embedding的生成过程

##### (1) $t_{1}(operator)$ 

![t1_output](material/t1.png)

$left \_\ childs:$ 

![](http://latex.codecogs.com/gif.latex?left\_childs)

![t1_left_childs](material/t1_l.png)

##### (2) $t_{2}(operator)$

![t2_output](material/t2.png)

$left _ childs:$ 

![t2_left_childs](material/t2_l.png)

##### (3) $t_{3}(operator)$

![t3_output](material/t3.png)

$left\_childs:$ 

![t3_left_childs](material/t3_l.png)

##### (4) $t_{4}(number)$

![t4_output](material/t4.png)

$left\_childs:$ 

![t4_left_childs](material/t4_l.png)

##### (5) $t_{5}(number)$

![t5_output](material/t5.png)

$left\_childs:$ 

![t5_left_childs](material/t5_l.png)

##### (6) $t_{6}(operator)$

![t6_output](material/t6.png)

$left\_childs:$ 

![t6_left_childs](material/t6_l.png)

##### (7) $t_{7}(number)$

![t7_output](material/t7.png)

$left\_childs:$ 

![t7_left_childs](material/t7_l.png)

##### (8) $t_{8}(number)$

![t8_output](material/t8.png)

$left\_childs:$ 

![t8_left_childs](material/t8_l.png)

##### (9) $t_{9}(operator)$

![t9_output](material/t9.png)

$left\_childs:$ 

![t9_left_childs](material/t9_l.png)

##### (10) $t_{10}(number)$

![t10_output](material/t10.png)

$left\_childs:$ 

![t10_left_childs](material/t10_l.png)

##### (11) $t_{11}(number)$

![t11_output](material/t11.png)

$left\_childs:$ 

![t11_left_childs](material/t11_l.png)
