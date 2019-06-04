class Statistic:
    """ class to keep data in """

    def __init__(self, loss_gen_train=None,
                 loss_dis_train=None,
                 loss_gen_val=None,
                 loss_dis_val=None,
                 dis_acc=None):

        self.loss_gen_train = loss_gen_train
        self.loss_dis_train = loss_dis_train
        self.loss_gen_val = loss_gen_val
        self.loss_dis_val = loss_dis_val
        self.dis_acc = dis_acc

        # todo: add more if you want
