??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02unknown8??
?
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:f *
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:f *
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
: *
dtype0
?
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:  *
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
: *
dtype0
?
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
: @*
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
:@*
dtype0
?
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:@@*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:@*
dtype0
?
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?* 
shared_nameconv3d_4/kernel
?
#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel*+
_output_shapes
:@?*
dtype0
s
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3d_4/bias
l
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes	
:?*
dtype0
?
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??* 
shared_nameconv3d_5/kernel
?
#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel*,
_output_shapes
:??*
dtype0
s
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3d_5/bias
l
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes	
:?*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f *%
shared_nameAdam/conv3d/kernel/m
?
(Adam/conv3d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/m**
_output_shapes
:f *
dtype0
|
Adam/conv3d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv3d/bias/m
u
&Adam/conv3d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv3d_1/kernel/m
?
*Adam/conv3d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/m**
_output_shapes
:  *
dtype0
?
Adam/conv3d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv3d_1/bias/m
y
(Adam/conv3d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv3d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv3d_2/kernel/m
?
*Adam/conv3d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/m**
_output_shapes
: @*
dtype0
?
Adam/conv3d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d_2/bias/m
y
(Adam/conv3d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv3d_3/kernel/m
?
*Adam/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/m**
_output_shapes
:@@*
dtype0
?
Adam/conv3d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d_3/bias/m
y
(Adam/conv3d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*'
shared_nameAdam/conv3d_4/kernel/m
?
*Adam/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/m*+
_output_shapes
:@?*
dtype0
?
Adam/conv3d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv3d_4/bias/m
z
(Adam/conv3d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*'
shared_nameAdam/conv3d_5/kernel/m
?
*Adam/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/m*,
_output_shapes
:??*
dtype0
?
Adam/conv3d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv3d_5/bias/m
z
(Adam/conv3d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_3/kernel/m
?
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f *%
shared_nameAdam/conv3d/kernel/v
?
(Adam/conv3d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/v**
_output_shapes
:f *
dtype0
|
Adam/conv3d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv3d/bias/v
u
&Adam/conv3d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv3d_1/kernel/v
?
*Adam/conv3d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/v**
_output_shapes
:  *
dtype0
?
Adam/conv3d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv3d_1/bias/v
y
(Adam/conv3d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv3d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv3d_2/kernel/v
?
*Adam/conv3d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/v**
_output_shapes
: @*
dtype0
?
Adam/conv3d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d_2/bias/v
y
(Adam/conv3d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv3d_3/kernel/v
?
*Adam/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/v**
_output_shapes
:@@*
dtype0
?
Adam/conv3d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d_3/bias/v
y
(Adam/conv3d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*'
shared_nameAdam/conv3d_4/kernel/v
?
*Adam/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/v*+
_output_shapes
:@?*
dtype0
?
Adam/conv3d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv3d_4/bias/v
z
(Adam/conv3d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*'
shared_nameAdam/conv3d_5/kernel/v
?
*Adam/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/v*,
_output_shapes
:??*
dtype0
?
Adam/conv3d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv3d_5/bias/v
z
(Adam/conv3d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_3/kernel/v
?
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv3d/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:f *(
shared_nameAdam/conv3d/kernel/vhat
?
+Adam/conv3d/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/vhat**
_output_shapes
:f *
dtype0
?
Adam/conv3d/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3d/bias/vhat
{
)Adam/conv3d/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/vhat*
_output_shapes
: *
dtype0
?
Adam/conv3d_1/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameAdam/conv3d_1/kernel/vhat
?
-Adam/conv3d_1/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/vhat**
_output_shapes
:  *
dtype0
?
Adam/conv3d_1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv3d_1/bias/vhat

+Adam/conv3d_1/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/vhat*
_output_shapes
: *
dtype0
?
Adam/conv3d_2/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/conv3d_2/kernel/vhat
?
-Adam/conv3d_2/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/vhat**
_output_shapes
: @*
dtype0
?
Adam/conv3d_2/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv3d_2/bias/vhat

+Adam/conv3d_2/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/vhat*
_output_shapes
:@*
dtype0
?
Adam/conv3d_3/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_nameAdam/conv3d_3/kernel/vhat
?
-Adam/conv3d_3/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/vhat**
_output_shapes
:@@*
dtype0
?
Adam/conv3d_3/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv3d_3/bias/vhat

+Adam/conv3d_3/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/vhat*
_output_shapes
:@*
dtype0
?
Adam/conv3d_4/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?**
shared_nameAdam/conv3d_4/kernel/vhat
?
-Adam/conv3d_4/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/vhat*+
_output_shapes
:@?*
dtype0
?
Adam/conv3d_4/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv3d_4/bias/vhat
?
+Adam/conv3d_4/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/vhat*
_output_shapes	
:?*
dtype0
?
Adam/conv3d_5/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??**
shared_nameAdam/conv3d_5/kernel/vhat
?
-Adam/conv3d_5/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/vhat*,
_output_shapes
:??*
dtype0
?
Adam/conv3d_5/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv3d_5/bias/vhat
?
+Adam/conv3d_5/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/vhat*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense/kernel/vhat
?
*Adam/dense/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/vhat* 
_output_shapes
:
??*
dtype0
?
Adam/dense/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense/bias/vhat
z
(Adam/dense/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense/bias/vhat*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameAdam/dense_1/kernel/vhat
?
,Adam/dense_1/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/vhat* 
_output_shapes
:
??*
dtype0
?
Adam/dense_1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/dense_1/bias/vhat
~
*Adam/dense_1/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/vhat*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameAdam/dense_2/kernel/vhat
?
,Adam/dense_2/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/vhat* 
_output_shapes
:
??*
dtype0
?
Adam/dense_2/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/dense_2/bias/vhat
~
*Adam/dense_2/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/vhat*
_output_shapes	
:?*
dtype0
?
Adam/dense_3/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdam/dense_3/kernel/vhat
?
,Adam/dense_3/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/vhat*
_output_shapes
:	?*
dtype0
?
Adam/dense_3/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3/bias/vhat
}
*Adam/dense_3/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/vhat*
_output_shapes
:*
dtype0

NoOpNoOp
Ѓ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
R
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
R
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
h

Okernel
Pbias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
h

Ukernel
Vbias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
h

[kernel
\bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
h

akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
?
giter

hbeta_1

ibeta_2
	jdecay
klearning_ratem?m?m?m?'m?(m?1m?2m?;m?<m?Em?Fm?Om?Pm?Um?Vm?[m?\m?am?bm?v?v?v?v?'v?(v?1v?2v?;v?<v?Ev?Fv?Ov?Pv?Uv?Vv?[v?\v?av?bv?vhat?vhat?vhat?vhat?'vhat?(vhat?1vhat?2vhat?;vhat?<vhat?Evhat?Fvhat?Ovhat?Pvhat?Uvhat?Vvhat?[vhat?\vhat?avhat?bvhat?
 
?
0
1
2
3
'4
(5
16
27
;8
<9
E10
F11
O12
P13
U14
V15
[16
\17
a18
b19
?
0
1
2
3
'4
(5
16
27
;8
<9
E10
F11
O12
P13
U14
V15
[16
\17
a18
b19
?
regularization_losses
llayer_metrics

mlayers
nmetrics
olayer_regularization_losses
trainable_variables
	variables
pnon_trainable_variables
 
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
qlayer_metrics

rlayers
smetrics
tlayer_regularization_losses
trainable_variables
	variables
unon_trainable_variables
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
vlayer_metrics

wlayers
xmetrics
ylayer_regularization_losses
 trainable_variables
!	variables
znon_trainable_variables
 
 
 
?
#regularization_losses
{layer_metrics

|layers
}metrics
~layer_regularization_losses
$trainable_variables
%	variables
non_trainable_variables
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
?
)regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
*trainable_variables
+	variables
?non_trainable_variables
 
 
 
?
-regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
.trainable_variables
/	variables
?non_trainable_variables
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
?
3regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
4trainable_variables
5	variables
?non_trainable_variables
 
 
 
?
7regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
8trainable_variables
9	variables
?non_trainable_variables
[Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
?
=regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
>trainable_variables
?	variables
?non_trainable_variables
 
 
 
?
Aregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Btrainable_variables
C	variables
?non_trainable_variables
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
?
Gregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Htrainable_variables
I	variables
?non_trainable_variables
 
 
 
?
Kregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Ltrainable_variables
M	variables
?non_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

O0
P1
?
Qregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Rtrainable_variables
S	variables
?non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

U0
V1
?
Wregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Xtrainable_variables
Y	variables
?non_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

[0
\1
?
]regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
^trainable_variables
_	variables
?non_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

a0
b1
?
cregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
dtrainable_variables
e	variables
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15

?0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
|z
VARIABLE_VALUEAdam/conv3d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d/kernel/vhatUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d/bias/vhatSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_1/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_1/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_2/kernel/vhatUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_2/bias/vhatSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_3/kernel/vhatUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_3/bias/vhatSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_4/kernel/vhatUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_4/bias/vhatSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv3d_5/kernel/vhatUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv3d_5/bias/vhatSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/dense/kernel/vhatUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense/bias/vhatSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_1/kernel/vhatUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_1/bias/vhatSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_2/kernel/vhatUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_2/bias/vhatSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_3/kernel/vhatUlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_3/bias/vhatSlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*6
_output_shapes$
": ????????????*
dtype0*+
shape": ????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias* 
Tin
2*
Tout
2*'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*/
config_proto

GPU

CPU2*0,1J 8*,
f'R%
#__inference_signature_wrapper_73819
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv3d/kernel/m/Read/ReadVariableOp&Adam/conv3d/bias/m/Read/ReadVariableOp*Adam/conv3d_1/kernel/m/Read/ReadVariableOp(Adam/conv3d_1/bias/m/Read/ReadVariableOp*Adam/conv3d_2/kernel/m/Read/ReadVariableOp(Adam/conv3d_2/bias/m/Read/ReadVariableOp*Adam/conv3d_3/kernel/m/Read/ReadVariableOp(Adam/conv3d_3/bias/m/Read/ReadVariableOp*Adam/conv3d_4/kernel/m/Read/ReadVariableOp(Adam/conv3d_4/bias/m/Read/ReadVariableOp*Adam/conv3d_5/kernel/m/Read/ReadVariableOp(Adam/conv3d_5/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp(Adam/conv3d/kernel/v/Read/ReadVariableOp&Adam/conv3d/bias/v/Read/ReadVariableOp*Adam/conv3d_1/kernel/v/Read/ReadVariableOp(Adam/conv3d_1/bias/v/Read/ReadVariableOp*Adam/conv3d_2/kernel/v/Read/ReadVariableOp(Adam/conv3d_2/bias/v/Read/ReadVariableOp*Adam/conv3d_3/kernel/v/Read/ReadVariableOp(Adam/conv3d_3/bias/v/Read/ReadVariableOp*Adam/conv3d_4/kernel/v/Read/ReadVariableOp(Adam/conv3d_4/bias/v/Read/ReadVariableOp*Adam/conv3d_5/kernel/v/Read/ReadVariableOp(Adam/conv3d_5/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp+Adam/conv3d/kernel/vhat/Read/ReadVariableOp)Adam/conv3d/bias/vhat/Read/ReadVariableOp-Adam/conv3d_1/kernel/vhat/Read/ReadVariableOp+Adam/conv3d_1/bias/vhat/Read/ReadVariableOp-Adam/conv3d_2/kernel/vhat/Read/ReadVariableOp+Adam/conv3d_2/bias/vhat/Read/ReadVariableOp-Adam/conv3d_3/kernel/vhat/Read/ReadVariableOp+Adam/conv3d_3/bias/vhat/Read/ReadVariableOp-Adam/conv3d_4/kernel/vhat/Read/ReadVariableOp+Adam/conv3d_4/bias/vhat/Read/ReadVariableOp-Adam/conv3d_5/kernel/vhat/Read/ReadVariableOp+Adam/conv3d_5/bias/vhat/Read/ReadVariableOp*Adam/dense/kernel/vhat/Read/ReadVariableOp(Adam/dense/bias/vhat/Read/ReadVariableOp,Adam/dense_1/kernel/vhat/Read/ReadVariableOp*Adam/dense_1/bias/vhat/Read/ReadVariableOp,Adam/dense_2/kernel/vhat/Read/ReadVariableOp*Adam/dense_2/bias/vhat/Read/ReadVariableOp,Adam/dense_3/kernel/vhat/Read/ReadVariableOp*Adam/dense_3/bias/vhat/Read/ReadVariableOpConst*d
Tin]
[2Y	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*'
f"R 
__inference__traced_save_74441
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv3d/kernel/mAdam/conv3d/bias/mAdam/conv3d_1/kernel/mAdam/conv3d_1/bias/mAdam/conv3d_2/kernel/mAdam/conv3d_2/bias/mAdam/conv3d_3/kernel/mAdam/conv3d_3/bias/mAdam/conv3d_4/kernel/mAdam/conv3d_4/bias/mAdam/conv3d_5/kernel/mAdam/conv3d_5/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv3d/kernel/vAdam/conv3d/bias/vAdam/conv3d_1/kernel/vAdam/conv3d_1/bias/vAdam/conv3d_2/kernel/vAdam/conv3d_2/bias/vAdam/conv3d_3/kernel/vAdam/conv3d_3/bias/vAdam/conv3d_4/kernel/vAdam/conv3d_4/bias/vAdam/conv3d_5/kernel/vAdam/conv3d_5/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/conv3d/kernel/vhatAdam/conv3d/bias/vhatAdam/conv3d_1/kernel/vhatAdam/conv3d_1/bias/vhatAdam/conv3d_2/kernel/vhatAdam/conv3d_2/bias/vhatAdam/conv3d_3/kernel/vhatAdam/conv3d_3/bias/vhatAdam/conv3d_4/kernel/vhatAdam/conv3d_4/bias/vhatAdam/conv3d_5/kernel/vhatAdam/conv3d_5/bias/vhatAdam/dense/kernel/vhatAdam/dense/bias/vhatAdam/dense_1/kernel/vhatAdam/dense_1/bias/vhatAdam/dense_2/kernel/vhatAdam/dense_2/bias/vhatAdam/dense_3/kernel/vhatAdam/dense_3/bias/vhat*c
Tin\
Z2X*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8**
f%R#
!__inference__traced_restore_74714??
?p
?	
 __inference__wrapped_model_73153
input_15
1littletimmy_conv3d_conv3d_readvariableop_resource6
2littletimmy_conv3d_biasadd_readvariableop_resource7
3littletimmy_conv3d_1_conv3d_readvariableop_resource8
4littletimmy_conv3d_1_biasadd_readvariableop_resource7
3littletimmy_conv3d_2_conv3d_readvariableop_resource8
4littletimmy_conv3d_2_biasadd_readvariableop_resource7
3littletimmy_conv3d_3_conv3d_readvariableop_resource8
4littletimmy_conv3d_3_biasadd_readvariableop_resource7
3littletimmy_conv3d_4_conv3d_readvariableop_resource8
4littletimmy_conv3d_4_biasadd_readvariableop_resource7
3littletimmy_conv3d_5_conv3d_readvariableop_resource8
4littletimmy_conv3d_5_biasadd_readvariableop_resource4
0littletimmy_dense_matmul_readvariableop_resource5
1littletimmy_dense_biasadd_readvariableop_resource6
2littletimmy_dense_1_matmul_readvariableop_resource7
3littletimmy_dense_1_biasadd_readvariableop_resource6
2littletimmy_dense_2_matmul_readvariableop_resource7
3littletimmy_dense_2_biasadd_readvariableop_resource6
2littletimmy_dense_3_matmul_readvariableop_resource7
3littletimmy_dense_3_biasadd_readvariableop_resource
identity??
(littleTimmy/conv3d/Conv3D/ReadVariableOpReadVariableOp1littletimmy_conv3d_conv3d_readvariableop_resource**
_output_shapes
:f *
dtype02*
(littleTimmy/conv3d/Conv3D/ReadVariableOp?
littleTimmy/conv3d/Conv3DConv3Dinput_10littleTimmy/conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingVALID*
strides	
f2
littleTimmy/conv3d/Conv3D?
)littleTimmy/conv3d/BiasAdd/ReadVariableOpReadVariableOp2littletimmy_conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)littleTimmy/conv3d/BiasAdd/ReadVariableOp?
littleTimmy/conv3d/BiasAddBiasAdd"littleTimmy/conv3d/Conv3D:output:01littleTimmy/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
littleTimmy/conv3d/BiasAdd?
littleTimmy/conv3d/ReluRelu#littleTimmy/conv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:??????????? 2
littleTimmy/conv3d/Relu?
*littleTimmy/conv3d_1/Conv3D/ReadVariableOpReadVariableOp3littletimmy_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02,
*littleTimmy/conv3d_1/Conv3D/ReadVariableOp?
littleTimmy/conv3d_1/Conv3DConv3D%littleTimmy/conv3d/Relu:activations:02littleTimmy/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingVALID*
strides	
2
littleTimmy/conv3d_1/Conv3D?
+littleTimmy/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp4littletimmy_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+littleTimmy/conv3d_1/BiasAdd/ReadVariableOp?
littleTimmy/conv3d_1/BiasAddBiasAdd$littleTimmy/conv3d_1/Conv3D:output:03littleTimmy/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
littleTimmy/conv3d_1/BiasAdd?
littleTimmy/conv3d_1/ReluRelu%littleTimmy/conv3d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:??????????? 2
littleTimmy/conv3d_1/Relu?
#littleTimmy/max_pooling3d/MaxPool3D	MaxPool3D'littleTimmy/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????DD *
ksize	
*
paddingVALID*
strides	
2%
#littleTimmy/max_pooling3d/MaxPool3D?
*littleTimmy/conv3d_2/Conv3D/ReadVariableOpReadVariableOp3littletimmy_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02,
*littleTimmy/conv3d_2/Conv3D/ReadVariableOp?
littleTimmy/conv3d_2/Conv3DConv3D,littleTimmy/max_pooling3d/MaxPool3D:output:02littleTimmy/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@*
paddingVALID*
strides	
2
littleTimmy/conv3d_2/Conv3D?
+littleTimmy/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp4littletimmy_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+littleTimmy/conv3d_2/BiasAdd/ReadVariableOp?
littleTimmy/conv3d_2/BiasAddBiasAdd$littleTimmy/conv3d_2/Conv3D:output:03littleTimmy/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@2
littleTimmy/conv3d_2/BiasAdd?
littleTimmy/conv3d_2/ReluRelu%littleTimmy/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????BB@2
littleTimmy/conv3d_2/Relu?
'littleTimmy/zero_padding3d/Pad/paddingsConst*
_output_shapes

:*
dtype0*A
value8B6"(                                    2)
'littleTimmy/zero_padding3d/Pad/paddings?
littleTimmy/zero_padding3d/PadPad'littleTimmy/conv3d_2/Relu:activations:00littleTimmy/zero_padding3d/Pad/paddings:output:0*
T0*3
_output_shapes!
:?????????DD@2 
littleTimmy/zero_padding3d/Pad?
*littleTimmy/conv3d_3/Conv3D/ReadVariableOpReadVariableOp3littletimmy_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02,
*littleTimmy/conv3d_3/Conv3D/ReadVariableOp?
littleTimmy/conv3d_3/Conv3DConv3D'littleTimmy/zero_padding3d/Pad:output:02littleTimmy/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@*
paddingVALID*
strides	
2
littleTimmy/conv3d_3/Conv3D?
+littleTimmy/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp4littletimmy_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+littleTimmy/conv3d_3/BiasAdd/ReadVariableOp?
littleTimmy/conv3d_3/BiasAddBiasAdd$littleTimmy/conv3d_3/Conv3D:output:03littleTimmy/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@2
littleTimmy/conv3d_3/BiasAdd?
littleTimmy/conv3d_3/ReluRelu%littleTimmy/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????BB@2
littleTimmy/conv3d_3/Relu?
%littleTimmy/max_pooling3d_1/MaxPool3D	MaxPool3D'littleTimmy/conv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????!!@*
ksize	
*
paddingVALID*
strides	
2'
%littleTimmy/max_pooling3d_1/MaxPool3D?
*littleTimmy/conv3d_4/Conv3D/ReadVariableOpReadVariableOp3littletimmy_conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02,
*littleTimmy/conv3d_4/Conv3D/ReadVariableOp?
littleTimmy/conv3d_4/Conv3DConv3D.littleTimmy/max_pooling3d_1/MaxPool3D:output:02littleTimmy/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingVALID*
strides	
2
littleTimmy/conv3d_4/Conv3D?
+littleTimmy/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp4littletimmy_conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+littleTimmy/conv3d_4/BiasAdd/ReadVariableOp?
littleTimmy/conv3d_4/BiasAddBiasAdd$littleTimmy/conv3d_4/Conv3D:output:03littleTimmy/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
littleTimmy/conv3d_4/BiasAdd?
littleTimmy/conv3d_4/ReluRelu%littleTimmy/conv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
littleTimmy/conv3d_4/Relu?
)littleTimmy/zero_padding3d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*A
value8B6"(                                    2+
)littleTimmy/zero_padding3d_1/Pad/paddings?
 littleTimmy/zero_padding3d_1/PadPad'littleTimmy/conv3d_4/Relu:activations:02littleTimmy/zero_padding3d_1/Pad/paddings:output:0*
T0*4
_output_shapes"
 :?????????!!?2"
 littleTimmy/zero_padding3d_1/Pad?
*littleTimmy/conv3d_5/Conv3D/ReadVariableOpReadVariableOp3littletimmy_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02,
*littleTimmy/conv3d_5/Conv3D/ReadVariableOp?
littleTimmy/conv3d_5/Conv3DConv3D)littleTimmy/zero_padding3d_1/Pad:output:02littleTimmy/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingVALID*
strides	
2
littleTimmy/conv3d_5/Conv3D?
+littleTimmy/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp4littletimmy_conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+littleTimmy/conv3d_5/BiasAdd/ReadVariableOp?
littleTimmy/conv3d_5/BiasAddBiasAdd$littleTimmy/conv3d_5/Conv3D:output:03littleTimmy/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
littleTimmy/conv3d_5/BiasAdd?
littleTimmy/conv3d_5/ReluRelu%littleTimmy/conv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
littleTimmy/conv3d_5/Relu?
;littleTimmy/global_average_pooling3d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2=
;littleTimmy/global_average_pooling3d/Mean/reduction_indices?
)littleTimmy/global_average_pooling3d/MeanMean'littleTimmy/conv3d_5/Relu:activations:0DlittleTimmy/global_average_pooling3d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2+
)littleTimmy/global_average_pooling3d/Mean?
'littleTimmy/dense/MatMul/ReadVariableOpReadVariableOp0littletimmy_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'littleTimmy/dense/MatMul/ReadVariableOp?
littleTimmy/dense/MatMulMatMul2littleTimmy/global_average_pooling3d/Mean:output:0/littleTimmy/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense/MatMul?
(littleTimmy/dense/BiasAdd/ReadVariableOpReadVariableOp1littletimmy_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(littleTimmy/dense/BiasAdd/ReadVariableOp?
littleTimmy/dense/BiasAddBiasAdd"littleTimmy/dense/MatMul:product:00littleTimmy/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense/BiasAdd?
littleTimmy/dense/ReluRelu"littleTimmy/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense/Relu?
)littleTimmy/dense_1/MatMul/ReadVariableOpReadVariableOp2littletimmy_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)littleTimmy/dense_1/MatMul/ReadVariableOp?
littleTimmy/dense_1/MatMulMatMul$littleTimmy/dense/Relu:activations:01littleTimmy/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense_1/MatMul?
*littleTimmy/dense_1/BiasAdd/ReadVariableOpReadVariableOp3littletimmy_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*littleTimmy/dense_1/BiasAdd/ReadVariableOp?
littleTimmy/dense_1/BiasAddBiasAdd$littleTimmy/dense_1/MatMul:product:02littleTimmy/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense_1/BiasAdd?
littleTimmy/dense_1/ReluRelu$littleTimmy/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense_1/Relu?
)littleTimmy/dense_2/MatMul/ReadVariableOpReadVariableOp2littletimmy_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)littleTimmy/dense_2/MatMul/ReadVariableOp?
littleTimmy/dense_2/MatMulMatMul&littleTimmy/dense_1/Relu:activations:01littleTimmy/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense_2/MatMul?
*littleTimmy/dense_2/BiasAdd/ReadVariableOpReadVariableOp3littletimmy_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*littleTimmy/dense_2/BiasAdd/ReadVariableOp?
littleTimmy/dense_2/BiasAddBiasAdd$littleTimmy/dense_2/MatMul:product:02littleTimmy/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense_2/BiasAdd?
littleTimmy/dense_2/ReluRelu$littleTimmy/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
littleTimmy/dense_2/Relu?
)littleTimmy/dense_3/MatMul/ReadVariableOpReadVariableOp2littletimmy_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)littleTimmy/dense_3/MatMul/ReadVariableOp?
littleTimmy/dense_3/MatMulMatMul&littleTimmy/dense_2/Relu:activations:01littleTimmy/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
littleTimmy/dense_3/MatMul?
*littleTimmy/dense_3/BiasAdd/ReadVariableOpReadVariableOp3littletimmy_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*littleTimmy/dense_3/BiasAdd/ReadVariableOp?
littleTimmy/dense_3/BiasAddBiasAdd$littleTimmy/dense_3/MatMul:product:02littleTimmy/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
littleTimmy/dense_3/BiasAdd?
littleTimmy/dense_3/SigmoidSigmoid$littleTimmy/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
littleTimmy/dense_3/Sigmoids
IdentityIdentitylittleTimmy/dense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????:::::::::::::::::::::_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?]
?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73901

inputs)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource+
'conv3d_5_conv3d_readvariableop_resource,
(conv3d_5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:f *
dtype02
conv3d/Conv3D/ReadVariableOp?
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingVALID*
strides	
f2
conv3d/Conv3D?
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3d/BiasAdd/ReadVariableOp?
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d/BiasAdd{
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d/Relu?
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02 
conv3d_1/Conv3D/ReadVariableOp?
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingVALID*
strides	
2
conv3d_1/Conv3D?
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_1/BiasAdd/ReadVariableOp?
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d_1/BiasAdd?
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d_1/Relu?
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????DD *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d/MaxPool3D?
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_2/Conv3D/ReadVariableOp?
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@*
paddingVALID*
strides	
2
conv3d_2/Conv3D?
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp?
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@2
conv3d_2/BiasAdd
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????BB@2
conv3d_2/Relu?
zero_padding3d/Pad/paddingsConst*
_output_shapes

:*
dtype0*A
value8B6"(                                    2
zero_padding3d/Pad/paddings?
zero_padding3d/PadPadconv3d_2/Relu:activations:0$zero_padding3d/Pad/paddings:output:0*
T0*3
_output_shapes!
:?????????DD@2
zero_padding3d/Pad?
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_3/Conv3D/ReadVariableOp?
conv3d_3/Conv3DConv3Dzero_padding3d/Pad:output:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@*
paddingVALID*
strides	
2
conv3d_3/Conv3D?
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp?
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@2
conv3d_3/BiasAdd
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????BB@2
conv3d_3/Relu?
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????!!@*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_1/MaxPool3D?
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02 
conv3d_4/Conv3D/ReadVariableOp?
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingVALID*
strides	
2
conv3d_4/Conv3D?
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp?
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
conv3d_4/BiasAdd?
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
conv3d_4/Relu?
zero_padding3d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*A
value8B6"(                                    2
zero_padding3d_1/Pad/paddings?
zero_padding3d_1/PadPadconv3d_4/Relu:activations:0&zero_padding3d_1/Pad/paddings:output:0*
T0*4
_output_shapes"
 :?????????!!?2
zero_padding3d_1/Pad?
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02 
conv3d_5/Conv3D/ReadVariableOp?
conv3d_5/Conv3DConv3Dzero_padding3d_1/Pad:output:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingVALID*
strides	
2
conv3d_5/Conv3D?
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_5/BiasAdd/ReadVariableOp?
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
conv3d_5/BiasAdd?
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
conv3d_5/Relu?
/global_average_pooling3d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/global_average_pooling3d/Mean/reduction_indices?
global_average_pooling3d/MeanMeanconv3d_5/Relu:activations:08global_average_pooling3d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
global_average_pooling3d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling3d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidg
IdentityIdentitydense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????:::::::::::::::::::::^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_73203

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
{
&__inference_conv3d_layer_call_fn_73175

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_731652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_conv3d_3_layer_call_fn_73266

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_732562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
L
0__inference_zero_padding3d_1_layer_call_fn_73313

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*T
fORM
K__inference_zero_padding3d_1_layer_call_and_return_conditional_losses_733072
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_73819
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*/
config_proto

GPU

CPU2*0,1J 8*)
f$R"
 __inference__wrapped_model_731532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_conv3d_layer_call_and_return_conditional_losses_73165

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:f *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? *
paddingVALID*
strides	
f2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2	
BiasAdd
ReluReluBiasAdd:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????:::v r
N
_output_shapes<
::8????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_conv3d_1_layer_call_fn_73197

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8???????????????????????????????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_731872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
g
K__inference_zero_padding3d_1_layer_call_and_return_conditional_losses_73307

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*A
value8B6"(                                    2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_littleTimmy_layer_call_fn_74028

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*/
config_proto

GPU

CPU2*0,1J 8*O
fJRH
F__inference_littleTimmy_layer_call_and_return_conditional_losses_736172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_conv3d_5_layer_call_fn_73335

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*O
_output_shapes=
;:9?????????????????????????????????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_733252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9?????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_73398

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?E
?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73496
input_1
conv3d_73352
conv3d_73354
conv3d_1_73357
conv3d_1_73359
conv3d_2_73363
conv3d_2_73365
conv3d_3_73369
conv3d_3_73371
conv3d_4_73375
conv3d_4_73377
conv3d_5_73381
conv3d_5_73383
dense_73409
dense_73411
dense_1_73436
dense_1_73438
dense_2_73463
dense_2_73465
dense_3_73490
dense_3_73492
identity??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_73352conv3d_73354*
Tin
2*
Tout
2*5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_731652 
conv3d/StatefulPartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_73357conv3d_1_73359*
Tin
2*
Tout
2*5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_731872"
 conv3d_1/StatefulPartitionedCall?
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????DD * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_732032
max_pooling3d/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_73363conv3d_2_73365*
Tin
2*
Tout
2*3
_output_shapes!
:?????????BB@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_732212"
 conv3d_2/StatefulPartitionedCall?
zero_padding3d/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????DD@* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*R
fMRK
I__inference_zero_padding3d_layer_call_and_return_conditional_losses_732382 
zero_padding3d/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall'zero_padding3d/PartitionedCall:output:0conv3d_3_73369conv3d_3_73371*
Tin
2*
Tout
2*3
_output_shapes!
:?????????BB@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_732562"
 conv3d_3/StatefulPartitionedCall?
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????!!@* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_732722!
max_pooling3d_1/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_73375conv3d_4_73377*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_732902"
 conv3d_4/StatefulPartitionedCall?
 zero_padding3d_1/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????!!?* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*T
fORM
K__inference_zero_padding3d_1_layer_call_and_return_conditional_losses_733072"
 zero_padding3d_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)zero_padding3d_1/PartitionedCall:output:0conv3d_5_73381conv3d_5_73383*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_733252"
 conv3d_5/StatefulPartitionedCall?
(global_average_pooling3d/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*\
fWRU
S__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_733422*
(global_average_pooling3d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling3d/PartitionedCall:output:0dense_73409dense_73411*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_733982
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73436dense_1_73438*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_734252!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73463dense_2_73465*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_734522!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73490dense_3_73492*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_734792!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_conv3d_2_layer_call_and_return_conditional_losses_73221

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd
ReluReluBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? :::v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_conv3d_5_layer_call_and_return_conditional_losses_73325

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*O
_output_shapes=
;:9?????????????????????????????????????*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*O
_output_shapes=
;:9?????????????????????????????????????2	
BiasAdd?
ReluReluBiasAdd:output:0*
T0*O
_output_shapes=
;:9?????????????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9?????????????????????????????????????:::w s
O
_output_shapes=
;:9?????????????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
o
S__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_73342

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv3d_1_layer_call_and_return_conditional_losses_73187

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? *
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2	
BiasAdd
ReluReluBiasAdd:output:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0*
T0*N
_output_shapes<
::8???????????????????????????????????? 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? :::v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
K
/__inference_max_pooling3d_1_layer_call_fn_73278

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_732722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?E
?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73617

inputs
conv3d_73561
conv3d_73563
conv3d_1_73566
conv3d_1_73568
conv3d_2_73572
conv3d_2_73574
conv3d_3_73578
conv3d_3_73580
conv3d_4_73584
conv3d_4_73586
conv3d_5_73590
conv3d_5_73592
dense_73596
dense_73598
dense_1_73601
dense_1_73603
dense_2_73606
dense_2_73608
dense_3_73611
dense_3_73613
identity??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_73561conv3d_73563*
Tin
2*
Tout
2*5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_731652 
conv3d/StatefulPartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_73566conv3d_1_73568*
Tin
2*
Tout
2*5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_731872"
 conv3d_1/StatefulPartitionedCall?
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????DD * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_732032
max_pooling3d/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_73572conv3d_2_73574*
Tin
2*
Tout
2*3
_output_shapes!
:?????????BB@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_732212"
 conv3d_2/StatefulPartitionedCall?
zero_padding3d/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????DD@* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*R
fMRK
I__inference_zero_padding3d_layer_call_and_return_conditional_losses_732382 
zero_padding3d/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall'zero_padding3d/PartitionedCall:output:0conv3d_3_73578conv3d_3_73580*
Tin
2*
Tout
2*3
_output_shapes!
:?????????BB@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_732562"
 conv3d_3/StatefulPartitionedCall?
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????!!@* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_732722!
max_pooling3d_1/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_73584conv3d_4_73586*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_732902"
 conv3d_4/StatefulPartitionedCall?
 zero_padding3d_1/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????!!?* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*T
fORM
K__inference_zero_padding3d_1_layer_call_and_return_conditional_losses_733072"
 zero_padding3d_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)zero_padding3d_1/PartitionedCall:output:0conv3d_5_73590conv3d_5_73592*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_733252"
 conv3d_5/StatefulPartitionedCall?
(global_average_pooling3d/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*\
fWRU
S__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_733422*
(global_average_pooling3d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling3d/PartitionedCall:output:0dense_73596dense_73598*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_733982
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73601dense_1_73603*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_734252!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73606dense_2_73608*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_734522!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73611dense_3_73613*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_734792!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_73425

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?E
?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73721

inputs
conv3d_73665
conv3d_73667
conv3d_1_73670
conv3d_1_73672
conv3d_2_73676
conv3d_2_73678
conv3d_3_73682
conv3d_3_73684
conv3d_4_73688
conv3d_4_73690
conv3d_5_73694
conv3d_5_73696
dense_73700
dense_73702
dense_1_73705
dense_1_73707
dense_2_73710
dense_2_73712
dense_3_73715
dense_3_73717
identity??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_73665conv3d_73667*
Tin
2*
Tout
2*5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_731652 
conv3d/StatefulPartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_73670conv3d_1_73672*
Tin
2*
Tout
2*5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_731872"
 conv3d_1/StatefulPartitionedCall?
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????DD * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_732032
max_pooling3d/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_73676conv3d_2_73678*
Tin
2*
Tout
2*3
_output_shapes!
:?????????BB@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_732212"
 conv3d_2/StatefulPartitionedCall?
zero_padding3d/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????DD@* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*R
fMRK
I__inference_zero_padding3d_layer_call_and_return_conditional_losses_732382 
zero_padding3d/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall'zero_padding3d/PartitionedCall:output:0conv3d_3_73682conv3d_3_73684*
Tin
2*
Tout
2*3
_output_shapes!
:?????????BB@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_732562"
 conv3d_3/StatefulPartitionedCall?
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????!!@* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_732722!
max_pooling3d_1/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_73688conv3d_4_73690*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_732902"
 conv3d_4/StatefulPartitionedCall?
 zero_padding3d_1/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????!!?* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*T
fORM
K__inference_zero_padding3d_1_layer_call_and_return_conditional_losses_733072"
 zero_padding3d_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)zero_padding3d_1/PartitionedCall:output:0conv3d_5_73694conv3d_5_73696*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_733252"
 conv3d_5/StatefulPartitionedCall?
(global_average_pooling3d/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*\
fWRU
S__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_733422*
(global_average_pooling3d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling3d/PartitionedCall:output:0dense_73700dense_73702*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_733982
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73705dense_1_73707*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_734252!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73710dense_2_73712*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_734522!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73715dense_3_73717*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_734792!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?E
?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73555
input_1
conv3d_73499
conv3d_73501
conv3d_1_73504
conv3d_1_73506
conv3d_2_73510
conv3d_2_73512
conv3d_3_73516
conv3d_3_73518
conv3d_4_73522
conv3d_4_73524
conv3d_5_73528
conv3d_5_73530
dense_73534
dense_73536
dense_1_73539
dense_1_73541
dense_2_73544
dense_2_73546
dense_3_73549
dense_3_73551
identity??conv3d/StatefulPartitionedCall? conv3d_1/StatefulPartitionedCall? conv3d_2/StatefulPartitionedCall? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? conv3d_5/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_73499conv3d_73501*
Tin
2*
Tout
2*5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_731652 
conv3d/StatefulPartitionedCall?
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_73504conv3d_1_73506*
Tin
2*
Tout
2*5
_output_shapes#
!:??????????? *$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_731872"
 conv3d_1/StatefulPartitionedCall?
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????DD * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_732032
max_pooling3d/PartitionedCall?
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_73510conv3d_2_73512*
Tin
2*
Tout
2*3
_output_shapes!
:?????????BB@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_732212"
 conv3d_2/StatefulPartitionedCall?
zero_padding3d/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????DD@* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*R
fMRK
I__inference_zero_padding3d_layer_call_and_return_conditional_losses_732382 
zero_padding3d/PartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall'zero_padding3d/PartitionedCall:output:0conv3d_3_73516conv3d_3_73518*
Tin
2*
Tout
2*3
_output_shapes!
:?????????BB@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_732562"
 conv3d_3/StatefulPartitionedCall?
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:?????????!!@* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_732722!
max_pooling3d_1/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_73522conv3d_4_73524*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_732902"
 conv3d_4/StatefulPartitionedCall?
 zero_padding3d_1/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????!!?* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*T
fORM
K__inference_zero_padding3d_1_layer_call_and_return_conditional_losses_733072"
 zero_padding3d_1/PartitionedCall?
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)zero_padding3d_1/PartitionedCall:output:0conv3d_5_73528conv3d_5_73530*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_733252"
 conv3d_5/StatefulPartitionedCall?
(global_average_pooling3d/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*\
fWRU
S__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_733422*
(global_average_pooling3d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling3d/PartitionedCall:output:0dense_73534dense_73536*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_733982
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73539dense_1_73541*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_734252!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73544dense_2_73546*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_734522!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73549dense_3_73551*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_734792!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_74124

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_conv3d_3_layer_call_and_return_conditional_losses_73256

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2	
BiasAdd
ReluReluBiasAdd:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@:::v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_2_layer_call_and_return_conditional_losses_73452

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_max_pooling3d_layer_call_fn_73209

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_732032
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_littleTimmy_layer_call_fn_73660
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*/
config_proto

GPU

CPU2*0,1J 8*O
fJRH
F__inference_littleTimmy_layer_call_and_return_conditional_losses_736172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_zero_padding3d_layer_call_and_return_conditional_losses_73238

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*A
value8B6"(                                    2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv3d_4_layer_call_and_return_conditional_losses_73290

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*O
_output_shapes=
;:9?????????????????????????????????????*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*O
_output_shapes=
;:9?????????????????????????????????????2	
BiasAdd?
ReluReluBiasAdd:output:0*
T0*O
_output_shapes=
;:9?????????????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@:::v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_74084

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
T
8__inference_global_average_pooling3d_layer_call_fn_73348

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*\
fWRU
S__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_733422
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
z
%__inference_dense_layer_call_fn_74093

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_733982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_dense_1_layer_call_fn_74113

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_734252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_dense_3_layer_call_fn_74153

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_734792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_73272

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv3d_2_layer_call_fn_73231

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8????????????????????????????????????@*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_732212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8????????????????????????????????????@2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8???????????????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8???????????????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_zero_padding3d_layer_call_fn_73244

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2*0,1J 8*R
fMRK
I__inference_zero_padding3d_layer_call_and_return_conditional_losses_732382
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_74104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_littleTimmy_layer_call_fn_73764
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*/
config_proto

GPU

CPU2*0,1J 8*O
fJRH
F__inference_littleTimmy_layer_call_and_return_conditional_losses_737212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
6
_output_shapes$
": ????????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_littleTimmy_layer_call_fn_74073

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*/
config_proto

GPU

CPU2*0,1J 8*O
fJRH
F__inference_littleTimmy_layer_call_and_return_conditional_losses_737212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_74144

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_conv3d_4_layer_call_fn_73300

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*O
_output_shapes=
;:9?????????????????????????????????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_732902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*O
_output_shapes=
;:9?????????????????????????????????????2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8????????????????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8????????????????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
۴
?#
__inference__traced_save_74441
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv3d_kernel_m_read_readvariableop1
-savev2_adam_conv3d_bias_m_read_readvariableop5
1savev2_adam_conv3d_1_kernel_m_read_readvariableop3
/savev2_adam_conv3d_1_bias_m_read_readvariableop5
1savev2_adam_conv3d_2_kernel_m_read_readvariableop3
/savev2_adam_conv3d_2_bias_m_read_readvariableop5
1savev2_adam_conv3d_3_kernel_m_read_readvariableop3
/savev2_adam_conv3d_3_bias_m_read_readvariableop5
1savev2_adam_conv3d_4_kernel_m_read_readvariableop3
/savev2_adam_conv3d_4_bias_m_read_readvariableop5
1savev2_adam_conv3d_5_kernel_m_read_readvariableop3
/savev2_adam_conv3d_5_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop3
/savev2_adam_conv3d_kernel_v_read_readvariableop1
-savev2_adam_conv3d_bias_v_read_readvariableop5
1savev2_adam_conv3d_1_kernel_v_read_readvariableop3
/savev2_adam_conv3d_1_bias_v_read_readvariableop5
1savev2_adam_conv3d_2_kernel_v_read_readvariableop3
/savev2_adam_conv3d_2_bias_v_read_readvariableop5
1savev2_adam_conv3d_3_kernel_v_read_readvariableop3
/savev2_adam_conv3d_3_bias_v_read_readvariableop5
1savev2_adam_conv3d_4_kernel_v_read_readvariableop3
/savev2_adam_conv3d_4_bias_v_read_readvariableop5
1savev2_adam_conv3d_5_kernel_v_read_readvariableop3
/savev2_adam_conv3d_5_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop6
2savev2_adam_conv3d_kernel_vhat_read_readvariableop4
0savev2_adam_conv3d_bias_vhat_read_readvariableop8
4savev2_adam_conv3d_1_kernel_vhat_read_readvariableop6
2savev2_adam_conv3d_1_bias_vhat_read_readvariableop8
4savev2_adam_conv3d_2_kernel_vhat_read_readvariableop6
2savev2_adam_conv3d_2_bias_vhat_read_readvariableop8
4savev2_adam_conv3d_3_kernel_vhat_read_readvariableop6
2savev2_adam_conv3d_3_bias_vhat_read_readvariableop8
4savev2_adam_conv3d_4_kernel_vhat_read_readvariableop6
2savev2_adam_conv3d_4_bias_vhat_read_readvariableop8
4savev2_adam_conv3d_5_kernel_vhat_read_readvariableop6
2savev2_adam_conv3d_5_bias_vhat_read_readvariableop5
1savev2_adam_dense_kernel_vhat_read_readvariableop3
/savev2_adam_dense_bias_vhat_read_readvariableop7
3savev2_adam_dense_1_kernel_vhat_read_readvariableop5
1savev2_adam_dense_1_bias_vhat_read_readvariableop7
3savev2_adam_dense_2_kernel_vhat_read_readvariableop5
1savev2_adam_dense_2_bias_vhat_read_readvariableop7
3savev2_adam_dense_3_kernel_vhat_read_readvariableop5
1savev2_adam_dense_3_bias_vhat_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_93930af8b7c74eea8abb40f584a2c176/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?3
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*?2
value?2B?2WB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*?
value?B?WB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv3d_kernel_m_read_readvariableop-savev2_adam_conv3d_bias_m_read_readvariableop1savev2_adam_conv3d_1_kernel_m_read_readvariableop/savev2_adam_conv3d_1_bias_m_read_readvariableop1savev2_adam_conv3d_2_kernel_m_read_readvariableop/savev2_adam_conv3d_2_bias_m_read_readvariableop1savev2_adam_conv3d_3_kernel_m_read_readvariableop/savev2_adam_conv3d_3_bias_m_read_readvariableop1savev2_adam_conv3d_4_kernel_m_read_readvariableop/savev2_adam_conv3d_4_bias_m_read_readvariableop1savev2_adam_conv3d_5_kernel_m_read_readvariableop/savev2_adam_conv3d_5_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop/savev2_adam_conv3d_kernel_v_read_readvariableop-savev2_adam_conv3d_bias_v_read_readvariableop1savev2_adam_conv3d_1_kernel_v_read_readvariableop/savev2_adam_conv3d_1_bias_v_read_readvariableop1savev2_adam_conv3d_2_kernel_v_read_readvariableop/savev2_adam_conv3d_2_bias_v_read_readvariableop1savev2_adam_conv3d_3_kernel_v_read_readvariableop/savev2_adam_conv3d_3_bias_v_read_readvariableop1savev2_adam_conv3d_4_kernel_v_read_readvariableop/savev2_adam_conv3d_4_bias_v_read_readvariableop1savev2_adam_conv3d_5_kernel_v_read_readvariableop/savev2_adam_conv3d_5_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop2savev2_adam_conv3d_kernel_vhat_read_readvariableop0savev2_adam_conv3d_bias_vhat_read_readvariableop4savev2_adam_conv3d_1_kernel_vhat_read_readvariableop2savev2_adam_conv3d_1_bias_vhat_read_readvariableop4savev2_adam_conv3d_2_kernel_vhat_read_readvariableop2savev2_adam_conv3d_2_bias_vhat_read_readvariableop4savev2_adam_conv3d_3_kernel_vhat_read_readvariableop2savev2_adam_conv3d_3_bias_vhat_read_readvariableop4savev2_adam_conv3d_4_kernel_vhat_read_readvariableop2savev2_adam_conv3d_4_bias_vhat_read_readvariableop4savev2_adam_conv3d_5_kernel_vhat_read_readvariableop2savev2_adam_conv3d_5_bias_vhat_read_readvariableop1savev2_adam_dense_kernel_vhat_read_readvariableop/savev2_adam_dense_bias_vhat_read_readvariableop3savev2_adam_dense_1_kernel_vhat_read_readvariableop1savev2_adam_dense_1_bias_vhat_read_readvariableop3savev2_adam_dense_2_kernel_vhat_read_readvariableop1savev2_adam_dense_2_bias_vhat_read_readvariableop3savev2_adam_dense_3_kernel_vhat_read_readvariableop1savev2_adam_dense_3_bias_vhat_read_readvariableop"/device:CPU:0*
_output_shapes
 *e
dtypes[
Y2W	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :f : :  : : @:@:@@:@:@?:?:??:?:
??:?:
??:?:
??:?:	?:: : : : : : : :f : :  : : @:@:@@:@:@?:?:??:?:
??:?:
??:?:
??:?:	?::f : :  : : @:@:@@:@:@?:?:??:?:
??:?:
??:?:
??:?:	?::f : :  : : @:@:@@:@:@?:?:??:?:
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:f : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
: @: 

_output_shapes
:@:0,
*
_output_shapes
:@@: 

_output_shapes
:@:1	-
+
_output_shapes
:@?:!


_output_shapes	
:?:2.
,
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
:f : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0 ,
*
_output_shapes
: @: !

_output_shapes
:@:0",
*
_output_shapes
:@@: #

_output_shapes
:@:1$-
+
_output_shapes
:@?:!%

_output_shapes	
:?:2&.
,
_output_shapes
:??:!'

_output_shapes	
:?:&("
 
_output_shapes
:
??:!)

_output_shapes	
:?:&*"
 
_output_shapes
:
??:!+

_output_shapes	
:?:&,"
 
_output_shapes
:
??:!-

_output_shapes	
:?:%.!

_output_shapes
:	?: /

_output_shapes
::00,
*
_output_shapes
:f : 1

_output_shapes
: :02,
*
_output_shapes
:  : 3

_output_shapes
: :04,
*
_output_shapes
: @: 5

_output_shapes
:@:06,
*
_output_shapes
:@@: 7

_output_shapes
:@:18-
+
_output_shapes
:@?:!9

_output_shapes	
:?:2:.
,
_output_shapes
:??:!;

_output_shapes	
:?:&<"
 
_output_shapes
:
??:!=

_output_shapes	
:?:&>"
 
_output_shapes
:
??:!?

_output_shapes	
:?:&@"
 
_output_shapes
:
??:!A

_output_shapes	
:?:%B!

_output_shapes
:	?: C

_output_shapes
::0D,
*
_output_shapes
:f : E

_output_shapes
: :0F,
*
_output_shapes
:  : G

_output_shapes
: :0H,
*
_output_shapes
: @: I

_output_shapes
:@:0J,
*
_output_shapes
:@@: K

_output_shapes
:@:1L-
+
_output_shapes
:@?:!M

_output_shapes	
:?:2N.
,
_output_shapes
:??:!O

_output_shapes	
:?:&P"
 
_output_shapes
:
??:!Q

_output_shapes	
:?:&R"
 
_output_shapes
:
??:!S

_output_shapes	
:?:&T"
 
_output_shapes
:
??:!U

_output_shapes	
:?:%V!

_output_shapes
:	?: W

_output_shapes
::X

_output_shapes
: 
?
|
'__inference_dense_2_layer_call_fn_74133

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2*0,1J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_734522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?-
!__inference__traced_restore_74714
file_prefix"
assignvariableop_conv3d_kernel"
assignvariableop_1_conv3d_bias&
"assignvariableop_2_conv3d_1_kernel$
 assignvariableop_3_conv3d_1_bias&
"assignvariableop_4_conv3d_2_kernel$
 assignvariableop_5_conv3d_2_bias&
"assignvariableop_6_conv3d_3_kernel$
 assignvariableop_7_conv3d_3_bias&
"assignvariableop_8_conv3d_4_kernel$
 assignvariableop_9_conv3d_4_bias'
#assignvariableop_10_conv3d_5_kernel%
!assignvariableop_11_conv3d_5_bias$
 assignvariableop_12_dense_kernel"
assignvariableop_13_dense_bias&
"assignvariableop_14_dense_1_kernel$
 assignvariableop_15_dense_1_bias&
"assignvariableop_16_dense_2_kernel$
 assignvariableop_17_dense_2_bias&
"assignvariableop_18_dense_3_kernel$
 assignvariableop_19_dense_3_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate
assignvariableop_25_total
assignvariableop_26_count,
(assignvariableop_27_adam_conv3d_kernel_m*
&assignvariableop_28_adam_conv3d_bias_m.
*assignvariableop_29_adam_conv3d_1_kernel_m,
(assignvariableop_30_adam_conv3d_1_bias_m.
*assignvariableop_31_adam_conv3d_2_kernel_m,
(assignvariableop_32_adam_conv3d_2_bias_m.
*assignvariableop_33_adam_conv3d_3_kernel_m,
(assignvariableop_34_adam_conv3d_3_bias_m.
*assignvariableop_35_adam_conv3d_4_kernel_m,
(assignvariableop_36_adam_conv3d_4_bias_m.
*assignvariableop_37_adam_conv3d_5_kernel_m,
(assignvariableop_38_adam_conv3d_5_bias_m+
'assignvariableop_39_adam_dense_kernel_m)
%assignvariableop_40_adam_dense_bias_m-
)assignvariableop_41_adam_dense_1_kernel_m+
'assignvariableop_42_adam_dense_1_bias_m-
)assignvariableop_43_adam_dense_2_kernel_m+
'assignvariableop_44_adam_dense_2_bias_m-
)assignvariableop_45_adam_dense_3_kernel_m+
'assignvariableop_46_adam_dense_3_bias_m,
(assignvariableop_47_adam_conv3d_kernel_v*
&assignvariableop_48_adam_conv3d_bias_v.
*assignvariableop_49_adam_conv3d_1_kernel_v,
(assignvariableop_50_adam_conv3d_1_bias_v.
*assignvariableop_51_adam_conv3d_2_kernel_v,
(assignvariableop_52_adam_conv3d_2_bias_v.
*assignvariableop_53_adam_conv3d_3_kernel_v,
(assignvariableop_54_adam_conv3d_3_bias_v.
*assignvariableop_55_adam_conv3d_4_kernel_v,
(assignvariableop_56_adam_conv3d_4_bias_v.
*assignvariableop_57_adam_conv3d_5_kernel_v,
(assignvariableop_58_adam_conv3d_5_bias_v+
'assignvariableop_59_adam_dense_kernel_v)
%assignvariableop_60_adam_dense_bias_v-
)assignvariableop_61_adam_dense_1_kernel_v+
'assignvariableop_62_adam_dense_1_bias_v-
)assignvariableop_63_adam_dense_2_kernel_v+
'assignvariableop_64_adam_dense_2_bias_v-
)assignvariableop_65_adam_dense_3_kernel_v+
'assignvariableop_66_adam_dense_3_bias_v/
+assignvariableop_67_adam_conv3d_kernel_vhat-
)assignvariableop_68_adam_conv3d_bias_vhat1
-assignvariableop_69_adam_conv3d_1_kernel_vhat/
+assignvariableop_70_adam_conv3d_1_bias_vhat1
-assignvariableop_71_adam_conv3d_2_kernel_vhat/
+assignvariableop_72_adam_conv3d_2_bias_vhat1
-assignvariableop_73_adam_conv3d_3_kernel_vhat/
+assignvariableop_74_adam_conv3d_3_bias_vhat1
-assignvariableop_75_adam_conv3d_4_kernel_vhat/
+assignvariableop_76_adam_conv3d_4_bias_vhat1
-assignvariableop_77_adam_conv3d_5_kernel_vhat/
+assignvariableop_78_adam_conv3d_5_bias_vhat.
*assignvariableop_79_adam_dense_kernel_vhat,
(assignvariableop_80_adam_dense_bias_vhat0
,assignvariableop_81_adam_dense_1_kernel_vhat.
*assignvariableop_82_adam_dense_1_bias_vhat0
,assignvariableop_83_adam_dense_2_kernel_vhat.
*assignvariableop_84_adam_dense_2_bias_vhat0
,assignvariableop_85_adam_dense_3_kernel_vhat.
*assignvariableop_86_adam_dense_3_bias_vhat
identity_88??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_9?	RestoreV2?RestoreV2_1?3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*?2
value?2B?2WB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*?
value?B?WB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*e
dtypes[
Y2W	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_3_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_3_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_4_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_4_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv3d_5_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv3d_5_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_1_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_1_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_2_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_2_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_3_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_3_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0	*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv3d_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv3d_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv3d_1_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv3d_1_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv3d_2_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv3d_2_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv3d_3_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv3d_3_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv3d_4_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv3d_4_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv3d_5_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv3d_5_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_dense_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_2_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_2_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_3_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_3_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv3d_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_conv3d_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv3d_1_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv3d_1_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv3d_2_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv3d_2_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv3d_3_kernel_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv3d_3_bias_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv3d_4_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv3d_4_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv3d_5_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv3d_5_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_dense_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp%assignvariableop_60_adam_dense_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_1_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_1_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_dense_2_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_dense_2_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp)assignvariableop_65_adam_dense_3_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp'assignvariableop_66_adam_dense_3_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv3d_kernel_vhatIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv3d_bias_vhatIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp-assignvariableop_69_adam_conv3d_1_kernel_vhatIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_conv3d_1_bias_vhatIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp-assignvariableop_71_adam_conv3d_2_kernel_vhatIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_conv3d_2_bias_vhatIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp-assignvariableop_73_adam_conv3d_3_kernel_vhatIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_conv3d_3_bias_vhatIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp-assignvariableop_75_adam_conv3d_4_kernel_vhatIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp+assignvariableop_76_adam_conv3d_4_bias_vhatIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp-assignvariableop_77_adam_conv3d_5_kernel_vhatIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp+assignvariableop_78_adam_conv3d_5_bias_vhatIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_kernel_vhatIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_bias_vhatIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_dense_1_kernel_vhatIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_dense_1_bias_vhatIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82_
Identity_83IdentityRestoreV2:tensors:83*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp,assignvariableop_83_adam_dense_2_kernel_vhatIdentity_83:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_83_
Identity_84IdentityRestoreV2:tensors:84*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp*assignvariableop_84_adam_dense_2_bias_vhatIdentity_84:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_84_
Identity_85IdentityRestoreV2:tensors:85*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp,assignvariableop_85_adam_dense_3_kernel_vhatIdentity_85:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_85_
Identity_86IdentityRestoreV2:tensors:86*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp*assignvariableop_86_adam_dense_3_bias_vhatIdentity_86:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_86?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_87Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_87?
Identity_88IdentityIdentity_87:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_88"#
identity_88Identity_88:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: :L

_output_shapes
: :M

_output_shapes
: :N

_output_shapes
: :O

_output_shapes
: :P

_output_shapes
: :Q

_output_shapes
: :R

_output_shapes
: :S

_output_shapes
: :T

_output_shapes
: :U

_output_shapes
: :V

_output_shapes
: :W

_output_shapes
: 
?]
?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73983

inputs)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource+
'conv3d_5_conv3d_readvariableop_resource,
(conv3d_5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:f *
dtype02
conv3d/Conv3D/ReadVariableOp?
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingVALID*
strides	
f2
conv3d/Conv3D?
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3d/BiasAdd/ReadVariableOp?
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d/BiasAdd{
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d/Relu?
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype02 
conv3d_1/Conv3D/ReadVariableOp?
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? *
paddingVALID*
strides	
2
conv3d_1/Conv3D?
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_1/BiasAdd/ReadVariableOp?
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d_1/BiasAdd?
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:??????????? 2
conv3d_1/Relu?
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:?????????DD *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d/MaxPool3D?
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_2/Conv3D/ReadVariableOp?
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@*
paddingVALID*
strides	
2
conv3d_2/Conv3D?
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp?
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@2
conv3d_2/BiasAdd
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????BB@2
conv3d_2/Relu?
zero_padding3d/Pad/paddingsConst*
_output_shapes

:*
dtype0*A
value8B6"(                                    2
zero_padding3d/Pad/paddings?
zero_padding3d/PadPadconv3d_2/Relu:activations:0$zero_padding3d/Pad/paddings:output:0*
T0*3
_output_shapes!
:?????????DD@2
zero_padding3d/Pad?
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype02 
conv3d_3/Conv3D/ReadVariableOp?
conv3d_3/Conv3DConv3Dzero_padding3d/Pad:output:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@*
paddingVALID*
strides	
2
conv3d_3/Conv3D?
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp?
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????BB@2
conv3d_3/BiasAdd
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????BB@2
conv3d_3/Relu?
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????!!@*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_1/MaxPool3D?
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype02 
conv3d_4/Conv3D/ReadVariableOp?
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingVALID*
strides	
2
conv3d_4/Conv3D?
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp?
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
conv3d_4/BiasAdd?
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
conv3d_4/Relu?
zero_padding3d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*A
value8B6"(                                    2
zero_padding3d_1/Pad/paddings?
zero_padding3d_1/PadPadconv3d_4/Relu:activations:0&zero_padding3d_1/Pad/paddings:output:0*
T0*4
_output_shapes"
 :?????????!!?2
zero_padding3d_1/Pad?
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype02 
conv3d_5/Conv3D/ReadVariableOp?
conv3d_5/Conv3DConv3Dzero_padding3d_1/Pad:output:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingVALID*
strides	
2
conv3d_5/Conv3D?
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3d_5/BiasAdd/ReadVariableOp?
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????2
conv3d_5/BiasAdd?
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????2
conv3d_5/Relu?
/global_average_pooling3d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/global_average_pooling3d/Mean/reduction_indices?
global_average_pooling3d/MeanMeanconv3d_5/Relu:activations:08global_average_pooling3d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
global_average_pooling3d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling3d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidg
IdentityIdentitydense_3/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r: ????????????:::::::::::::::::::::^ Z
6
_output_shapes$
": ????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_73479

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
J
input_1?
serving_default_input_1:0 ????????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"ŋ
_tf_keras_model??{"class_name": "Model", "name": "littleTimmy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "littleTimmy", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 140, 140, 2350, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 102]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 102]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 1]}, "data_format": "channels_last"}, "name": "max_pooling3d", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["max_pooling3d", 0, 0, {}]]]}, {"class_name": "ZeroPadding3D", "config": {"name": "zero_padding3d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "name": "zero_padding3d", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["zero_padding3d", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 1]}, "data_format": "channels_last"}, "name": "max_pooling3d_1", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["max_pooling3d_1", 0, 0, {}]]]}, {"class_name": "ZeroPadding3D", "config": {"name": "zero_padding3d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "name": "zero_padding3d_1", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_5", "inbound_nodes": [[["zero_padding3d_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling3D", "config": {"name": "global_average_pooling3d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling3d", "inbound_nodes": [[["conv3d_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling3d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 140, 140, 2350, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "littleTimmy", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 140, 140, 2350, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 102]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 102]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 1]}, "data_format": "channels_last"}, "name": "max_pooling3d", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["max_pooling3d", 0, 0, {}]]]}, {"class_name": "ZeroPadding3D", "config": {"name": "zero_padding3d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "name": "zero_padding3d", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["zero_padding3d", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 1]}, "data_format": "channels_last"}, "name": "max_pooling3d_1", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["max_pooling3d_1", 0, 0, {}]]]}, {"class_name": "ZeroPadding3D", "config": {"name": "zero_padding3d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "name": "zero_padding3d_1", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_5", "inbound_nodes": [[["zero_padding3d_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling3D", "config": {"name": "global_average_pooling3d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling3d", "inbound_nodes": [[["conv3d_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling3d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_3", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00039999998989515007, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": true}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 140, 140, 2350, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 140, 140, 2350, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 102]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 102]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 140, 140, 2350, 1]}}
?	

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 138, 138, 23, 32]}}
?
#regularization_losses
$trainable_variables
%	variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling3D", "name": "max_pooling3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 68, 22, 32]}}
?
-regularization_losses
.trainable_variables
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding3D", "name": "zero_padding3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "zero_padding3d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 68, 21, 64]}}
?
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling3D", "name": "max_pooling3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 20, 64]}}
?
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding3D", "name": "zero_padding3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "zero_padding3d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 19, 128]}}
?
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GlobalAveragePooling3D", "name": "global_average_pooling3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "global_average_pooling3d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

Okernel
Pbias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

Ukernel
Vbias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

[kernel
\bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
giter

hbeta_1

ibeta_2
	jdecay
klearning_ratem?m?m?m?'m?(m?1m?2m?;m?<m?Em?Fm?Om?Pm?Um?Vm?[m?\m?am?bm?v?v?v?v?'v?(v?1v?2v?;v?<v?Ev?Fv?Ov?Pv?Uv?Vv?[v?\v?av?bv?vhat?vhat?vhat?vhat?'vhat?(vhat?1vhat?2vhat?;vhat?<vhat?Evhat?Fvhat?Ovhat?Pvhat?Uvhat?Vvhat?[vhat?\vhat?avhat?bvhat?"
	optimizer
 "
trackable_list_wrapper
?
0
1
2
3
'4
(5
16
27
;8
<9
E10
F11
O12
P13
U14
V15
[16
\17
a18
b19"
trackable_list_wrapper
?
0
1
2
3
'4
(5
16
27
;8
<9
E10
F11
O12
P13
U14
V15
[16
\17
a18
b19"
trackable_list_wrapper
?
regularization_losses
llayer_metrics

mlayers
nmetrics
olayer_regularization_losses
trainable_variables
	variables
pnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
+:)f 2conv3d/kernel
: 2conv3d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
qlayer_metrics

rlayers
smetrics
tlayer_regularization_losses
trainable_variables
	variables
unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+  2conv3d_1/kernel
: 2conv3d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
vlayer_metrics

wlayers
xmetrics
ylayer_regularization_losses
 trainable_variables
!	variables
znon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#regularization_losses
{layer_metrics

|layers
}metrics
~layer_regularization_losses
$trainable_variables
%	variables
non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+ @2conv3d_2/kernel
:@2conv3d_2/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
)regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
*trainable_variables
+	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
.trainable_variables
/	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+@@2conv3d_3/kernel
:@2conv3d_3/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
3regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
4trainable_variables
5	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
8trainable_variables
9	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,@?2conv3d_4/kernel
:?2conv3d_4/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
=regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
>trainable_variables
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Aregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Btrainable_variables
C	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2conv3d_5/kernel
:?2conv3d_5/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
Gregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Htrainable_variables
I	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Kregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Ltrainable_variables
M	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
??2dense/kernel
:?2
dense/bias
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
Qregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Rtrainable_variables
S	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_1/kernel
:?2dense_1/bias
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
Wregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
Xtrainable_variables
Y	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:?2dense_2/bias
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
]regularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
^trainable_variables
_	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
?
cregularization_losses
?layer_metrics
?layers
?metrics
 ?layer_regularization_losses
dtrainable_variables
e	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
0:.f 2Adam/conv3d/kernel/m
: 2Adam/conv3d/bias/m
2:0  2Adam/conv3d_1/kernel/m
 : 2Adam/conv3d_1/bias/m
2:0 @2Adam/conv3d_2/kernel/m
 :@2Adam/conv3d_2/bias/m
2:0@@2Adam/conv3d_3/kernel/m
 :@2Adam/conv3d_3/bias/m
3:1@?2Adam/conv3d_4/kernel/m
!:?2Adam/conv3d_4/bias/m
4:2??2Adam/conv3d_5/kernel/m
!:?2Adam/conv3d_5/bias/m
%:#
??2Adam/dense/kernel/m
:?2Adam/dense/bias/m
':%
??2Adam/dense_1/kernel/m
 :?2Adam/dense_1/bias/m
':%
??2Adam/dense_2/kernel/m
 :?2Adam/dense_2/bias/m
&:$	?2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
0:.f 2Adam/conv3d/kernel/v
: 2Adam/conv3d/bias/v
2:0  2Adam/conv3d_1/kernel/v
 : 2Adam/conv3d_1/bias/v
2:0 @2Adam/conv3d_2/kernel/v
 :@2Adam/conv3d_2/bias/v
2:0@@2Adam/conv3d_3/kernel/v
 :@2Adam/conv3d_3/bias/v
3:1@?2Adam/conv3d_4/kernel/v
!:?2Adam/conv3d_4/bias/v
4:2??2Adam/conv3d_5/kernel/v
!:?2Adam/conv3d_5/bias/v
%:#
??2Adam/dense/kernel/v
:?2Adam/dense/bias/v
':%
??2Adam/dense_1/kernel/v
 :?2Adam/dense_1/bias/v
':%
??2Adam/dense_2/kernel/v
 :?2Adam/dense_2/bias/v
&:$	?2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
3:1f 2Adam/conv3d/kernel/vhat
!: 2Adam/conv3d/bias/vhat
5:3  2Adam/conv3d_1/kernel/vhat
#:! 2Adam/conv3d_1/bias/vhat
5:3 @2Adam/conv3d_2/kernel/vhat
#:!@2Adam/conv3d_2/bias/vhat
5:3@@2Adam/conv3d_3/kernel/vhat
#:!@2Adam/conv3d_3/bias/vhat
6:4@?2Adam/conv3d_4/kernel/vhat
$:"?2Adam/conv3d_4/bias/vhat
7:5??2Adam/conv3d_5/kernel/vhat
$:"?2Adam/conv3d_5/bias/vhat
(:&
??2Adam/dense/kernel/vhat
!:?2Adam/dense/bias/vhat
*:(
??2Adam/dense_1/kernel/vhat
#:!?2Adam/dense_1/bias/vhat
*:(
??2Adam/dense_2/kernel/vhat
#:!?2Adam/dense_2/bias/vhat
):'	?2Adam/dense_3/kernel/vhat
": 2Adam/dense_3/bias/vhat
?2?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73496
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73983
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73901
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73555?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_73153?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *5?2
0?-
input_1 ????????????
?2?
+__inference_littleTimmy_layer_call_fn_74028
+__inference_littleTimmy_layer_call_fn_73764
+__inference_littleTimmy_layer_call_fn_74073
+__inference_littleTimmy_layer_call_fn_73660?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_conv3d_layer_call_and_return_conditional_losses_73165?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8????????????????????????????????????
?2?
&__inference_conv3d_layer_call_fn_73175?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8????????????????????????????????????
?2?
C__inference_conv3d_1_layer_call_and_return_conditional_losses_73187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8???????????????????????????????????? 
?2?
(__inference_conv3d_1_layer_call_fn_73197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8???????????????????????????????????? 
?2?
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_73203?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
-__inference_max_pooling3d_layer_call_fn_73209?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
C__inference_conv3d_2_layer_call_and_return_conditional_losses_73221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8???????????????????????????????????? 
?2?
(__inference_conv3d_2_layer_call_fn_73231?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8???????????????????????????????????? 
?2?
I__inference_zero_padding3d_layer_call_and_return_conditional_losses_73238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
.__inference_zero_padding3d_layer_call_fn_73244?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
C__inference_conv3d_3_layer_call_and_return_conditional_losses_73256?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8????????????????????????????????????@
?2?
(__inference_conv3d_3_layer_call_fn_73266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8????????????????????????????????????@
?2?
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_73272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
/__inference_max_pooling3d_1_layer_call_fn_73278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
C__inference_conv3d_4_layer_call_and_return_conditional_losses_73290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8????????????????????????????????????@
?2?
(__inference_conv3d_4_layer_call_fn_73300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *D?A
??<8????????????????????????????????????@
?2?
K__inference_zero_padding3d_1_layer_call_and_return_conditional_losses_73307?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
0__inference_zero_padding3d_1_layer_call_fn_73313?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
C__inference_conv3d_5_layer_call_and_return_conditional_losses_73325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *E?B
@?=9?????????????????????????????????????
?2?
(__inference_conv3d_5_layer_call_fn_73335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *E?B
@?=9?????????????????????????????????????
?2?
S__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_73342?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
8__inference_global_average_pooling3d_layer_call_fn_73348?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
@__inference_dense_layer_call_and_return_conditional_losses_74084?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_74093?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_74104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_74113?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_74124?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_74133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_74144?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_74153?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2B0
#__inference_signature_wrapper_73819input_1?
 __inference__wrapped_model_73153?'(12;<EFOPUV[\ab??<
5?2
0?-
input_1 ????????????
? "1?.
,
dense_3!?
dense_3??????????
C__inference_conv3d_1_layer_call_and_return_conditional_losses_73187?V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "L?I
B??
08???????????????????????????????????? 
? ?
(__inference_conv3d_1_layer_call_fn_73197?V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "??<8???????????????????????????????????? ?
C__inference_conv3d_2_layer_call_and_return_conditional_losses_73221?'(V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "L?I
B??
08????????????????????????????????????@
? ?
(__inference_conv3d_2_layer_call_fn_73231?'(V?S
L?I
G?D
inputs8???????????????????????????????????? 
? "??<8????????????????????????????????????@?
C__inference_conv3d_3_layer_call_and_return_conditional_losses_73256?12V?S
L?I
G?D
inputs8????????????????????????????????????@
? "L?I
B??
08????????????????????????????????????@
? ?
(__inference_conv3d_3_layer_call_fn_73266?12V?S
L?I
G?D
inputs8????????????????????????????????????@
? "??<8????????????????????????????????????@?
C__inference_conv3d_4_layer_call_and_return_conditional_losses_73290?;<V?S
L?I
G?D
inputs8????????????????????????????????????@
? "M?J
C?@
09?????????????????????????????????????
? ?
(__inference_conv3d_4_layer_call_fn_73300?;<V?S
L?I
G?D
inputs8????????????????????????????????????@
? "@?=9??????????????????????????????????????
C__inference_conv3d_5_layer_call_and_return_conditional_losses_73325?EFW?T
M?J
H?E
inputs9?????????????????????????????????????
? "M?J
C?@
09?????????????????????????????????????
? ?
(__inference_conv3d_5_layer_call_fn_73335?EFW?T
M?J
H?E
inputs9?????????????????????????????????????
? "@?=9??????????????????????????????????????
A__inference_conv3d_layer_call_and_return_conditional_losses_73165?V?S
L?I
G?D
inputs8????????????????????????????????????
? "L?I
B??
08???????????????????????????????????? 
? ?
&__inference_conv3d_layer_call_fn_73175?V?S
L?I
G?D
inputs8????????????????????????????????????
? "??<8???????????????????????????????????? ?
B__inference_dense_1_layer_call_and_return_conditional_losses_74104^UV0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_1_layer_call_fn_74113QUV0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_2_layer_call_and_return_conditional_losses_74124^[\0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_2_layer_call_fn_74133Q[\0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_3_layer_call_and_return_conditional_losses_74144]ab0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_3_layer_call_fn_74153Pab0?-
&?#
!?
inputs??????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_74084^OP0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
%__inference_dense_layer_call_fn_74093QOP0?-
&?#
!?
inputs??????????
? "????????????
S__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_73342?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
8__inference_global_average_pooling3d_layer_call_fn_73348?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "!????????????????????
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73496?'(12;<EFOPUV[\abG?D
=?:
0?-
input_1 ????????????
p

 
? "%?"
?
0?????????
? ?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73555?'(12;<EFOPUV[\abG?D
=?:
0?-
input_1 ????????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73901?'(12;<EFOPUV[\abF?C
<?9
/?,
inputs ????????????
p

 
? "%?"
?
0?????????
? ?
F__inference_littleTimmy_layer_call_and_return_conditional_losses_73983?'(12;<EFOPUV[\abF?C
<?9
/?,
inputs ????????????
p 

 
? "%?"
?
0?????????
? ?
+__inference_littleTimmy_layer_call_fn_73660y'(12;<EFOPUV[\abG?D
=?:
0?-
input_1 ????????????
p

 
? "???????????
+__inference_littleTimmy_layer_call_fn_73764y'(12;<EFOPUV[\abG?D
=?:
0?-
input_1 ????????????
p 

 
? "???????????
+__inference_littleTimmy_layer_call_fn_74028x'(12;<EFOPUV[\abF?C
<?9
/?,
inputs ????????????
p

 
? "???????????
+__inference_littleTimmy_layer_call_fn_74073x'(12;<EFOPUV[\abF?C
<?9
/?,
inputs ????????????
p 

 
? "???????????
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_73272?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
/__inference_max_pooling3d_1_layer_call_fn_73278?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_73203?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
-__inference_max_pooling3d_layer_call_fn_73209?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
#__inference_signature_wrapper_73819?'(12;<EFOPUV[\abJ?G
? 
@?=
;
input_10?-
input_1 ????????????"1?.
,
dense_3!?
dense_3??????????
K__inference_zero_padding3d_1_layer_call_and_return_conditional_losses_73307?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
0__inference_zero_padding3d_1_layer_call_fn_73313?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
I__inference_zero_padding3d_layer_call_and_return_conditional_losses_73238?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
.__inference_zero_padding3d_layer_call_fn_73244?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA?????????????????????????????????????????????