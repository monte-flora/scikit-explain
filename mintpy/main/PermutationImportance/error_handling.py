"""There are a handful of different errors and warnings that we can report. This
houses all of them and provides information regarding ways to fix them."""


class InvalidStrategyException(Exception):
    """Thrown when a scoring strategy is invalid"""

    def __init__(self, strategy, msg=None, options=None):
        if msg is None:
            msg = (
                "%s is not a valid strategy for determining the optimal variable. "
                % strategy
            )
            msg += "\nShould be a callable or a valid string option. "
            if options is not None:
                msg += "Valid options are\n%r" % options

        super(InvalidStrategyException, self).__init__(msg)
        self.strategy = strategy
        self.options = None


class InvalidInputException(Exception):
    """Thrown when the input to the program does not match expectations"""

    def __init__(self, value, msg=None):
        if msg is None:
            msg = "Input value does not match expectations: %s" % value

        super(InvalidInputException, self).__init__(msg)
        self.value = value


class InvalidDataException(Exception):
    """Thrown when the training or scoring data is not of the right type"""

    def __init__(self, data, msg=None):
        if msg is None:
            msg = "Data is not of the right format"

        super(InvalidDataException, self).__init__(msg)
        self.data = data


class UnmatchedLengthPredictionsException(Exception):
    """Thrown when the number of predictions doesn't match truths"""

    def __init__(self, truths, predictions, msg=None):
        if msg is None:
            msg = "Shapes of truths and predictions do not match: %r and %r" % (
                truths.shape,
                predictions.shape,
            )

        super(UnmatchedLengthPredictionsException, self).__init__(msg)
        self.truths = truths
        self.predictions = predictions


class UnmatchingProbabilisticForecastsException(Exception):
    """Thrown when the shape of probabilisic predictions doesn't match the truths"""

    def __init__(self, truths, predictions, msg=None):
        if msg is None:
            msg = "Shapes of truths and predictions do not match: %r and %r" % (
                truths.shape,
                predictions.shape,
            )

        super(UnmatchingProbabilisticForecastsException, self).__init__(msg)
        self.truths = truths
        self.predictions = predictions


class AmbiguousProbabilisticForecastsException(Exception):
    """Thrown when classes were not provided for converting probabilistic
    predictions to deterministic ones but are required"""

    def __init__(self, truths, predictions, msg=None):
        if msg is None:
            msg = "Classes not provided for converting probabilistic predictions to deterministic ones"

        super(AmbiguousProbabilisticForecastsException, self).__init__(msg)
        self.truths = truths
        self.predictions = predictions


class FullImportanceResultWarning(Warning):
    """Thrown when we try to add a result to a full
    :class:`PermutationImportance.result.ImportanceResult`"""

    pass
