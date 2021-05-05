import numpy as np

class СlassificationTree:
    def __init__(self):
        pass
#         self.splits = {}
        
    def bin_entropy(self, sorted_target):
        rate = sum(sorted_target)/len(sorted_target)
        if rate in [0, 1]:
            return sum([-rate, -(1-rate)])
        else:
            return sum([-rate*np.log2(rate), -(1-rate)*np.log2((1-rate))])
    
    def splitter(self, x, y):
        x_board = [None, None]
        max_ig = 0
        for i in range(x.shape[1]):
            temp_arr = np.array([np.array([x[:, i], y]).T[q] for q in np.argsort(x[:, i])])
            unique_x = np.unique(x[:, i])
            try:
                half_step = min(unique_x[1:] - unique_x[:-1])/2
            except ValueError:
                continue

            for u in unique_x[:-1]:
                mask = temp_arr[:, 0] <= u+half_step
                sorted_target_true = temp_arr[mask][:, 1]
                sorted_target_false = temp_arr[~mask][:, 1]
                ig = self.bin_entropy(temp_arr[:, 1]) - ((len(temp_arr[mask])/len(temp_arr))*self.bin_entropy(sorted_target_true) +\
                                             (len(temp_arr[~mask])/len(temp_arr))*self.bin_entropy(sorted_target_false))
                if ig > max_ig:
                    x_board[0], x_board[1] = i, u+half_step
                    max_ig = ig
        return x_board
    
    def tree_ways(self, x, y, links = np.array([[0, 0]]), n = 0):
        k, board = self.splitter(x, y)
        self.splits[n] = (k, board)
        if k == board == None:
            self.splits[n] = len((y == 1).nonzero()[0])/len(y)
            return links

        mask = x[:, k] <= board
        inds_true = (mask).nonzero()[0]
        inds_false = (~mask).nonzero()[0]

        x_true = np.take(x, inds_true, axis=0)
        y_true = np.take(y, inds_true)
        x_false = np.take(x, inds_false, axis=0)
        y_false = np.take(y, inds_false)

        if inds_true.size != 0:
            links = np.vstack((links, [n, max(links[:, 1])+1]))
            links = self.tree_ways(x_true, y_true, links, max(links[:, 1]))
        if inds_false.size != 0:
            links = np.vstack((links, [n, max(links[:, 1])+1]))
            links = self.tree_ways(x_false, y_false, links, max(links[:, 1]))
        return links
    
    def fit(self, *args, **kwargs):
        self.splits = {}
        self.links = self.tree_ways(*args, **kwargs)[1:]
        
    def get_proba_obj(self, obj):
        start_node = 0
        while True:
            try:
                var, board = self.splits[start_node]
            except:
                return self.splits[start_node]
            node_indxs = (self.links[:, 0] == start_node).nonzero()[0]
            if obj[var] <= board:
                start_node = np.take(self.links, node_indxs, axis=0)[0, 1]
            else:
                start_node = np.take(self.links, node_indxs, axis=0)[1, 1]
    
    def get_proba(self, x):
        return np.array([self.get_proba_obj(obj) for obj in x])