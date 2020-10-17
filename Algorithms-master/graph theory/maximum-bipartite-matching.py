
class GFG:
    def __init__(self, graph):

        # residual graph
        self.graph = graph
        self.ppl = len(graph)
        self.jobs = len(graph[0])
    
    '''
    a dfs-based recursive function that returns true if a matching for vertex u is possible
    '''
    def bpm(self, u, matchR, seen):

        # try every job one by one
        for v in range(self.jobs):
            # if application u is interested in job v and v is not seen
            if self.graph[u][v] and seen[v] == False:
                # mark v as visited
                seen[v] = True
                '''
                if job 'v' is not assigned to an applicant OR previously assigned applicant
                for job v (which is matchR[v]) has an alternative job available
                Since v is marked as visited in the above line, matchR[v] in the following recursive
                call will not get job 'v' again
                '''
                if matchR[v] == -1 or self.bpm(matchR[v], matchR, seen):
                    matchR[v] = u
                    return True
        return False

    # returns maximum number of matching
    def maxBPM(self):
        '''
        an array to keep track of the applicants assigned to jobs
        the value of matchR[i] is the applicant number assigned to job i,
        the value -1 indicates nobody is assigned
        '''
        matchR = [-1] * self.jobs

        # count of jobs assigned to applicants
        result = 0
        for i in range(self.ppl):
            # mark all jobs assigned to applicants
            seen = [False] * self.jobs

            # find if the applicant 'u' can get a job
            if self.bpm(i, matchR, seen):
                result += 1
        return result

if __name__ == "__main__":
    bpGraph =[[0, 1, 1, 0, 0, 0], 
              [1, 0, 0, 1, 0, 0], 
              [0, 0, 1, 0, 0, 0], 
              [0, 0, 1, 1, 0, 0], 
              [0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0, 1]]
    
    g = GFG(bpGraph)

    print("Maximum number of applicants that can get job is %d " % g.maxBPM())