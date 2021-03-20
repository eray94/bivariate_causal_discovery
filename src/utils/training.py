
class TrainingUtils:
    @staticmethod
    def start():
        print("**************************************************************")
        print("***************   TRAINING has been started!    **************")
        print("**************************************************************")

    @staticmethod
    def complete():
        print("***** TRAINING is completed and model weights are saved! *****")
        print("**************************************************************")

    @staticmethod
    def counter(index):
        print("Tuebingen Dataset {} is training...".format(index), end="\r")
