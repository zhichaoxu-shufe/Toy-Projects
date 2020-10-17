pdb debugging tutorial

To start use: `python -m pdb filename.py`

Some key commands:

Set break point: 
`break 10 # set the break point to the line 10 of this file`
`b filename.py: 20 # set the break point to line 20 of another file`

Delete break point: 

`b  # to look at all the previous points we set`
 `cl 2 # to delete the number 2 break point`

Run the file:

`n # next, run the single line`
`s #  step, get into the function`
`c # continue, jump to the next break point`

Examine:

`p parameter # print, print the value of variable`
`l # list, to view the code in the current line`
`a # to view all the variables in the stack`

Other commands:

`q # or exit, terminate and exit`
`r # or return, execute until return from current function`
`pp # print the value of a variable`
`help # to seek help`