from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
import random

class MRMultilineInput(MRJob):               
    def mapper_init(self):
        self.in_body = False
        self.body = []
        self.day = ''
        self.month = ''
        self.id = ''
    
    def mapper(self, _, line):
        line = line.strip()
        if line.find("Message-ID: ") != -1:
            self.id = line[line.find("<")+1:line.find(">")].lower()
        if line.find("Date: ") == 0:
            email_day = line[line.find(" ")+1:line.find(",")].strip()[:3].lower()
            self.day = email_day
            idx = line.find(", ")+4
            self.month = line[idx:idx+4].strip()
        if not line and not self.in_body:
            self.in_body=True
        
        l = len(line) - 1
        if l >= 0 and line.find('\"') == l and self.in_body:
            if self.day and self.body:
                yield None, f"{self.id};{self.day};{self.month};"+''.join(self.body)            
            self.body = []    
            self.in_body = False
            self.day = ''
            self.month=''
            self.id = ''
        
        if self.in_body:
            self.body.append(line)
                       
if __name__ == '__main__':
    MRMultilineInput.OUTPUT_PROTOCOL = RawValueProtocol
    MRMultilineInput.run()
