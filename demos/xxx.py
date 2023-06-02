# from multiprocessing import Process, Queue
# from time import sleep

# queue = Queue(maxsize=64)
# cqueue1 = Queue(maxsize=64)
# cqueue2 = Queue(maxsize=64)

# def torunA(queue, cqueue1, cqueue2):

#     # #while queue.qsize() >0 :
#     #     record = queue.get()
#     #     print(record)
#     #     sleep(1)
#     for _ in range(12500):
#         queue.put((3, 4.5))
#         cqueue1.put(45)
#         cqueue2.put(1)
#     print("worker closed")

# def torunB(queue, cqueue1, cqueue2):
#     x = []
#     for _ in range(12500):
#         x.append(queue.get())
#         x.append(cqueue1.get())
#         x.append(cqueue2.get())
#     print(len(x))


# if __name__ == '__main__':
#     processes = [Process(target=torunA, args=(queue, cqueue1, cqueue2,)), Process(target=torunB, args=(queue, cqueue1, cqueue2,))]

#     for process in processes:
#         process.start()
#         print('Process started')

#     for process in processes:
#         print(process.join())
#         print('done')

# #https://stackoverflow.com/questions/48363237/python-multiprocessing-queue-implementation

from squanch import *
from scipy.stats import unitary_group
import copy
import time
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import multiprocessing

def shor_encode(qsys):
    # psi is state to send, q1...q8 are ancillas from top to bottom in diagram
    psi, q1, q2, q3, q4, q5, q6, q7, q8 = qsys.qubits
    # Gates are enumerated left to right, top to bottom from figure
    CNOT(psi, q3)
    CNOT(psi, q6)
    H(psi)
    H(q3)
    H(q6)
    CNOT(psi, q1)
    CNOT(psi, q2) 
    CNOT(q3, q4)
    CNOT(q3, q5)
    CNOT(q6, q7)
    CNOT(q6, q8)
    return psi, q1, q2, q3, q4, q5, q6, q7, q8

def shor_decode(psi, q1, q2, q3, q4, q5, q6, q7, q8):
    # same enumeration as Alice
    CNOT(psi, q1)
    CNOT(psi, q2)
    TOFFOLI(q2, q1, psi)
    CNOT(q3, q4)
    CNOT(q3, q5)
    TOFFOLI(q5, q4, q3)
    CNOT(q6, q7)
    CNOT(q6, q8)
    TOFFOLI(q7, q8, q6) # Toffoli control qubit order doesn't matter
    H(psi)
    H(q3)
    H(q6)
    CNOT(psi, q3)
    CNOT(psi, q6)
    TOFFOLI(q6, q3, psi)
    return psi # psi is now Alice's original state


class Alice(Agent):
    '''Alice sends an error-protected state to Bob'''
    def run(self):
        for qsys in self.qstream:
            # send the encoded qubits to Bob 
            for qubit in shor_encode(qsys):
                self.qsend(bob, qubit)


class DumbAlice(Agent):
    '''DumbAlice sends an uncorrected state to DumbBob'''   
    def run(self):
        for qsys in self.qstream:
            for qubit in qsys.qubits:
                self.qsend(dumb_bob, qubit)

class Bob(Agent):
    '''Bob receives and error-corrects Alice's state'''
    def run(self):
        measurement_results = []
        for _ in self.qstream:
            # Bob receives 9 qubits representing Alice's encoded state
            received = [self.qrecv(alice) for _ in range(9)]
            # Decode and measure the original state
            psi_true = shor_decode(*received)
            measurement_results.append(psi_true.measure())
        self.output(measurement_results)

class DumbBob(Agent):
    '''DumbBob gets DumbAlice's non-corrected state'''
    def run(self):
        measurement_results = []
        for _ in self.qstream:
            received = [self.qrecv(dumb_alice) for _ in range(9)]
            psi_true = received[0]
            measurement_results.append(psi_true.measure())
        self.output(measurement_results)


class ShorError(QError):

    def __init__(self, qchannel):
        '''
        Instatiate the error model from the parent class
        :param QChannel qchannel: parent quantum channel
        '''
        QError.__init__(self, qchannel)
        self.count = 0 
        self.error_applied = False

    def apply(self, qubit):
        '''
        Apply a random unitary operation to one of the qubits in a set of 9
        :param Qubit qubit: qubit from quantum channel
        :return: either unchanged qubit or None
        '''
        # reset error for each group of 9 qubits
        if self.count == 0:
            self.error_applied = False
        self.count = (self.count + 1) % 9
        # qubit could be None if combining with other error models, such as attenuation
        if not self.error_applied and qubit is not None:
            if np.random.rand() < 0.5: # apply the error
                random_unitary = unitary_group.rvs(2) # pick a random U(2) matrix
                qubit.apply(random_unitary)
                self.error_applied = True
        return qubit
    

class ShorQChannel(QChannel):
    '''Represents a quantum channel with a Shor error applied'''
    
    def __init__(self, from_agent, to_agent):
        QChannel.__init__(self, from_agent, to_agent)
        # register the error model
        self.errors = [ShorError(self)] 

def to_bits(string):
    '''Convert a string to a list of bits'''
    result = []
    for c in string:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def from_bits(bits):
    '''Convert a list of bits to a string'''
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


multiprocessing.set_start_method("fork", force=True)

# Prepare a message to send
msg = "Peter Shor once lived in Ruddock 238! But who was Airman?"
bits = to_bits(msg)

# Encode the message as spin eigenstates
qstream = QStream(9, len(bits)) # 9 qubits per encoded state
for bit, qsystem in zip(bits, qstream):
    if bit == 1: 
        X(qsystem.qubit(0)) 

# Alice and Bob will use error correction
out = Agent.shared_output()
alice = Alice(qstream, out)
bob = Bob(qstream, out)
alice.qconnect(bob, ShorQChannel)

# Dumb agents won't use error correction
qstream2 = copy.deepcopy(qstream)
dumb_alice = DumbAlice(qstream2, out)
dumb_bob = DumbBob(qstream2, out)
dumb_alice.qconnect(dumb_bob, ShorQChannel)

# Run everything and record results
start_time = time.perf_counter()
Simulation(dumb_alice, dumb_bob, alice, bob).run(monitor_progress=False)
finish_time =time.perf_counter()

print("Total simulation runtime: ",  finish_time - start_time, "seconds")
print("DumbAlice sent:   {}".format(msg))
print("DumbBob received: {}\n".format(from_bits(out["DumbBob"])))
print("Alice sent:       {}".format(msg))
print("Bob received:     {}".format(from_bits(out["Bob"])))



