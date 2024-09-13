from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        output = []
        s = self.start_state
        for x in input_seq:
            print(self.transition_fn(s,x))
            s = self.transition_fn(s,x)
            output.append(self.output_fn(s))
        return output
        pass


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    #the first number of the tuple - current sum
    #the second number of the tuple - carry
    #we start with zero sum and zero carry
    start_state = [0,0] # Change
    
    
    def transition_fn(self, s, x):
        s_ = s.copy()
        #print(x)
        #print((x[0]+x[1]+s[1]) % 2)
        #sum is equal to digits' sum + carry mod 2
        s_[0] = (x[0]+x[1]+s[1]) % 2
        #carry is equal to 1 if digits' sum + carry is bigger or equal to 2
        #zero otherwise
        if((x[0]+x[1]+s[1])>=2):
            s_[1] = 1
        else:
            s_[1] = 0
        #print(s)
        return s_
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        return s[0]
        pass



class Reverser(SM):
    #the first element of the list - list of strings
    #the second element of the list - state
    #zero means - initial Nones
    #one means - reversed elements of the first list
    #two means - Nones for the second list
    #three means - finishing
    start_state = [[],0] # Change

    def transition_fn(self, s, x):
        res = s.copy()
        if(s[1]==0 and x!='end'):
            res[0].append(x)
            return res
        if(x=='end'):
            res[1] = 1
            return res
        if(len(res[0])==0):
            res[1] = 2
            return res
        return res
        # Your code here
        pass

    def output_fn(self, s):
        if (s[1]==0):
            return None
            #print('None')
        if(s[1]==1):
            return s[0].pop()
            #print(s[0].pop())
        
        # Your code here
        pass


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.f1 = f1
        self.f2 = f2
        self.start_state = np.zeros((Wss.shape[0],1))
        self.s = np.zeros((Wss.shape[0],1))
        #print((Wss.shape[0],1))
        # Your code here
        pass
    def transition_fn(self, s, x):
        print('s=',s)
        return self.f1(self.Wss@s+self.Wsx@x+self.Wss_0)
        # Your code here
        pass
    def output_fn(self, s):
        print(self.f2(self.Wo@s+self.Wo_0))
        return self.f2(self.Wo@s+self.Wo_0)
        # Your code here
        pass
