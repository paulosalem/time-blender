# Standard models
import numpy as np
import pandas as pd

from time_blender.coordination_events import PastEvent, CumulativeEvent, ParentValueEvent, TemporarySwitch, Choice, \
    SeasonalEvent
from time_blender.core import LambdaEvent, ConstantEvent, wrapped_constant_param, Event, Invariant
from time_blender.deterministic_events import WaveEvent, ClockEvent, WalkEvent, IdentityEvent, ClipEvent
from time_blender.random_events import NormalEvent, wrap_in_resistance, BernoulliEvent, PoissonEvent

from clize import Parameter


from time_blender.util import shift_weekend_and_holidays
from time_blender.cli import cli_model, a_list


class SimpleModels:
    @staticmethod
    @cli_model
    def cycle(base:float=10.0, period:float=72, growth_rate:float=2):

        period_fluctuation = WalkEvent(NormalEvent(0, 1), initial_pos=period, capture_parent_value=False)
        amplitude_trend = ClockEvent() * ConstantEvent(base) * NormalEvent(3, 1)
        we = WaveEvent(period_fluctuation, amplitude_trend)

        capacity_trend = ClockEvent() * ConstantEvent(growth_rate*base) * NormalEvent(3, 0.1)

        return we + capacity_trend


class ClassicModels:

    @staticmethod
    @cli_model
    def ar(p: int, *, constant: float=0, error_mean: float=0, error_std: float=1, coefs_low: float=-1, coefs_high: float=1,
           coefs: a_list=None, error_event: Parameter.IGNORE=None, capture_parent_value=True):
        """
        Creates a new AR (autoreressive) model. The model's coeficients can either be generated automatically
        by providing coefs_low and coefs_high parameters, or be explicitly defined by providing a list in the
        coefs parameter.

        :param p: The order of the AR model (how far back should it look).
        :param constant: The model's constant term.
        :param error_mean: The mean of the normal error component.
        :param error_std: The standard deviation of the normal error component.
        :param coefs_low: If specified, defines the lower bound of the coeficients to be generated. If left None,
                          then the coefs parameter must be specified.
        :param coefs_high: If specified, defines the upper bound of the coeficients to be generated. If left None,
                          then the coefs parameter must be specified.
        :param coefs: A list or dict with numeric keys of the coeficients to be employed. Must have size p. The i-th
                      element (if list) or key (if dict) correspond to the i-th coeficient.
                      If this is specified, coefs_low and coefs_high are ignored.
        :param error_event: An error event. If specified, it is used instead of error_mean and error_std.
        :capture_parent_value: Whether the parent value should be used as the new current value to which
                               the event's execution is added. This is useful to embed the present event
                               into larger contexts and accumulate on top of their feedback.
        :return: An AR model.
        """

        # check coeficients
        if (coefs is None) and (coefs_low is None) and (coefs_high is None):
            raise ValueError("Either coefs or coefs_loe, coefs_high must be specified.")

        # check error events
        if (coefs is error_mean) and (error_std is None) and (error_event is None):
            raise ValueError("Some error must be specified.")

        # Start with the model's constant.
        if error_event is not None:
            if not isinstance(constant, Event):
                x = ConstantEvent(constant, parallel_events=[error_event])
            else:
                x = constant.parallel_to(error_event)
        else:
            if not isinstance(constant, Event):
                x = ConstantEvent(constant)
            else:
                x = constant

        past = []

        # Add the autoregressive terms
        for i in range(0, p):

            if coefs is not None:
                alpha = coefs[i]
            else:
                alpha = np.random.uniform(coefs_low, coefs_high)

            pe = PastEvent(i + 1, allow_learning=False)
            past.append(pe)

            if error_event is not None:
                error = PastEvent(i, allow_learning=False)
                error.refers_to(error_event)
            else:
                error = NormalEvent(error_mean, error_std)

            x = x + pe * ConstantEvent(alpha) + error

        # connect past events to the series to which they refer to
        for pe in past:
            if capture_parent_value:
                pe.refers_to(ParentValueEvent(x))

            else:
                pe.refers_to(x)

        return x

    @staticmethod
    @cli_model
    def ma(q, *, series_mean: float=0, error_mean: float=0, error_std: float=1,
           coefs_low: float=-1, coefs_high: float=1, coefs: a_list=None, error_event: Parameter.IGNORE=None):
        """
        Creates a new MA (Moving Average) model. The model's coeficients can either be generated automatically
        by providing coefs_low and coefs_high parameters (default), or be explicitly defined by providing a list in the
        coefs parameter.

        :param q: The order of the MA model (how far back should it look).
        :param series_mean: The mean of the series.
        :param error_mean: The mean of the normal error of each past random shock.
        :param error_std: The standard deviation of the normal error of each past random shock.
        :param coefs_low: If specified, defines the lower bound of the coeficients to be generated. If left None,
                          then the coefs parameter must be specified.
        :param coefs_high: If specified, defines the upper bound of the coeficients to be generated. If left None,
                          then the coefs parameter must be specified.
        :param coefs: A list or dict with numeric keys of the coeficients to be employed. Must have size p. The i-th
                      element (if list) or key (if dict) correspond to the i-th coeficient.
                      If this is specified, coefs_low and coefs_high are ignored.
        :param error_event: An error event. If specified, it is used instead of error_mean and error_std.

        :return: The MA model.
        """

        # check coeficients
        if (coefs is None) and (coefs_low is None) and (coefs_high is None):
            raise ValueError("Either coefs or coefs_loe, coefs_high must be specified.")

        # error shocks
        if error_event is None:
            error_event = NormalEvent(error_mean, error_std)

        # Put the mean term first
        x = ConstantEvent(series_mean, parallel_events=[error_event])

        past = []

        # Add model terms
        for i in range(0, q):

            if coefs is not None:
                alpha = coefs[i]
            else:
                alpha = np.random.uniform(coefs_low, coefs_high)

            p = PastEvent(i + 1, allow_learning=False)
            past.append(p)
            x = x + p * ConstantEvent(alpha)

        # connect past events to the series to which they refer to
        for p in past:
            p.refers_to(error_event)

        return x

    @staticmethod
    @cli_model
    def arma(p, q, constant: float=0, error_mean: float=0, error_std: float=1,
             ar_coefs_low: float=-1, ar_coefs_high: float=1, ar_coefs: a_list=None,
             ma_coefs_low: float=-1, ma_coefs_high: float=1, ma_coefs: a_list=None,
             capture_parent_value=True):
        """
        Creates an ARMA model. This differs slightly from simply summing AR and MA models, because here a common
        normal error series is also provided

        :param p:
        :param q:
        :param constant:
        :param error_mean:
        :param error_std:
        :param ar_coefs_low:
        :param ar_coefs_high:
        :param ar_coefs:
        :param ma_coefs_low:
        :param ma_coefs_high:
        :param ma_coefs:
        :param capture_parent_value:
        :return:
        """

        # common error series
        error_event = NormalEvent(error_mean, error_std)

        m1 = ClassicModels.ar(p, constant=constant, coefs_low=ar_coefs_low, coefs_high=ar_coefs_high, coefs=ar_coefs,
                error_event=error_event, capture_parent_value=capture_parent_value)

        m2 = ClassicModels.ma(q, series_mean=0.0, coefs_low=ma_coefs_low, coefs_high=ma_coefs_high, coefs=ma_coefs,
                error_event=error_event)

        return m1 + m2

    @staticmethod
    @cli_model
    def arima(p, q, constant: float=0, error_mean: float=0, error_std: float=1,
             ar_coefs_low: float=-1, ar_coefs_high: float=1, ar_coefs: a_list=None,
             ma_coefs_low: float=-1, ma_coefs_high: float=1, ma_coefs: a_list=None,
             capture_parent_value=True):
        """
        Creates an ARIMA model. This adds the integration not found in ARMA. That is to say, values are accumulated
        over time.

        :param p:
        :param q:
        :param constant:
        :param error_mean:
        :param error_std:
        :param ar_coefs_low:
        :param ar_coefs_high:
        :param ar_coefs:
        :param ma_coefs_low:
        :param ma_coefs_high:
        :param ma_coefs:
        :param capture_parent_value:
        :return:
        """

        return CumulativeEvent(\
                    ClassicModels.arma(p=p, q=1, constant=constant, error_mean=error_mean, error_std=error_std,
                                       ar_coefs_low=ar_coefs_low, ar_coefs_high=ar_coefs_high, ar_coefs=ar_coefs,
                                       ma_coefs_low=ma_coefs_low, ma_coefs_high=ma_coefs_high, ma_coefs=ma_coefs,
                                       capture_parent_value=capture_parent_value))


class BankingModels:

    # TODO put newer version here

    @staticmethod
    @cli_model
    def salary_earner(salary_value=20000, payment_day: int=1, *,
                      regular_expense_mean=50, regular_expense_std=10,
                      large_expense_mean=1000, large_expense_std=100,
                      emergency_probability=0.1, emergencies_mean=500, emergencies_std=100):
        ##########################################
        # Job parameters
        ##########################################
        salary = ConstantEvent(salary_value, name='salary',
                               require_lower_bound=0.0,
                               learning_normal_std = 5000)

        payday = ConstantEvent(payment_day, name='payday', allow_learning=False)

        ########################################
        # Daily life
        ########################################
        typical_daily_cashflow = SeasonalEvent(salary, day=payday, fill_with_previous=False)

        # regular expenses
        typical_daily_cashflow -= ClipEvent(NormalEvent(regular_expense_mean, regular_expense_std),
                                            min_value=ConstantEvent(0.0, allow_learning=False))

        # large expenses
        typical_daily_cashflow -= ClipEvent(PoissonEvent(1)*NormalEvent(regular_expense_mean, regular_expense_std),
                                            min_value=ConstantEvent(0.0, allow_learning=False))

        # emergencies
        typical_daily_cashflow -= ClipEvent(BernoulliEvent(emergency_probability) * NormalEvent(emergencies_mean,
                                                                                                emergencies_std),
                                            min_value=ConstantEvent(0.0, allow_learning=False))

        # investments
        investment_base_mean = salary_value*0.025
        investment_base_std = salary_value*0.0012
        typical_daily_cashflow -= ClipEvent(Choice([ConstantEvent(0),
                                                    NormalEvent(investment_base_mean, investment_base_std),
                                                    NormalEvent(2*investment_base_mean, 2*investment_base_std),
                                                    NormalEvent(4*investment_base_mean, 4*investment_base_std)],
                                                   fix_choice=True),
                                            min_value=ConstantEvent(0.0, allow_learning=False))

        # cyclic residual
        typical_daily_cashflow -= WaveEvent(180, salary_value/100)

        non_payday_process = TemporarySwitch(typical_daily_cashflow,
                                             ConstantEvent(0.0, allow_learning=False),
                                             switch_duration=ConstantEvent(10, learning_normal_std=2))

        def aux_daily_cashflow(t, i, memory, sub_events):

            # ensures that on payday there is no temporary switch
            if t.day == sub_events['payday'].execute(t):
                flow = sub_events['typical_daily_cashflow'].execute(t)
            else:
                flow = sub_events['non_payday_process'].execute(t)

            return sub_events['parent_value'].execute(t) + flow


        parent_value = ParentValueEvent(default=0)

        x = LambdaEvent(aux_daily_cashflow, sub_events={'payday': payday,
                                                        'typical_daily_cashflow': typical_daily_cashflow,
                                                        'non_payday_process': non_payday_process,
                                                        'parent_value': parent_value})

        x = wrap_in_resistance(x,
                               top_resistance_levels=[3*salary_value],
                               bottom_resistance_levels=[-salary_value],
                               top_resistance_strength_event=ClipEvent(NormalEvent(0.1, 0.05), min_value=0.0),
                               bottom_resistance_strength_event=ClipEvent(NormalEvent(2, 1), min_value=0.0),
                               tolerance=salary_value,
                               top_resistance_probability=0.99,
                               bottom_resistance_probability=0.99)

        ##########################
        # Invariants
        ##########################

        invariant_1 = Invariant(lambda t, events: events['salary'].execute(t) >= 10000,
                                events={'salary': salary})

        x.add_invariant(invariant_1)

        salary_earner_model = x
        #salary_earner_model = Replicated(x,
        #                                 duration_per_replication=NormalEvent(mean=360, std=90, allow_learning=False),
        #                                 max_replication=3)

        return salary_earner_model


    @staticmethod
    @cli_model
    def salary_earner_simple(salary_value=5000, payment_day: int=1, *, expense_mean=100.0, expense_sd=300.0):

        # ensure we are working with a ConstantEvent
        salary = wrapped_constant_param(prefix='banking', name='salary', value=salary_value, require_lower_bound=0.0)

        # Daily expense model
        daily_normal_expense = ClipEvent(NormalEvent(expense_mean, expense_sd), min_value=0.0)

        def aux(t, i, memory, sub_events):
            t_next_month = t + pd.DateOffset(months=1)

            actual_payment_day_cur = shift_weekend_and_holidays(pd.Timestamp(t.year, t.month,
                                                                             sub_events['payment_day']),
                                                                 direction='backward')
            actual_payment_day_next = shift_weekend_and_holidays(pd.Timestamp(t_next_month.year, t_next_month.month,
                                                                              sub_events['payment_day']),
                                                                    direction='backward')

            if t.date() == actual_payment_day_cur:
                # current month
                memory['money'] = memory.get('money', 0.0) + sub_events['salary'].execute(t)

            elif t.date() == actual_payment_day_next:
                # advance for the next month, if applicable
                memory['money'] = memory.get('money', 0.0) + sub_events['salary'].execute(t)

            else:
                memory['money'] = memory.get('money', 0.0) - sub_events['daily_normal_expense'].execute(t)

            return memory['money']

        # The final model
        model = LambdaEvent(aux, sub_events={'salary': salary,
                                             'daily_normal_expense': daily_normal_expense,
                                             'payment_day': payment_day})
        return model


class EconomicModels:

    @staticmethod
    @cli_model
    def kondratiev_business_cycle(base: float = 0.0, growth_mean: float = 1, growth_sd: float = 2,
                             wave_period: float = 12, wave_amplitude: float= 0.05):
        """
        A naive interpretation of so-called "Kondratieve waves" business cycle theory.

        :param base: The initial economic condition.
        :param growth_mean: The mean of economic growth.
        :param growth_sd: The standard deviation of economic growth.
        :param wave_period: The period of the wave that modifies present conditions.
        :param wave_amplitude: The amplitude of the wave that modifies present conditions.
        :return:
        """
        we = WaveEvent(wave_period, wave_amplitude)
        return WalkEvent(NormalEvent(growth_mean, growth_sd), initial_pos=base) * (ConstantEvent(1) + we)


class EcologyModels:

    @staticmethod
    @cli_model
    def predator_prey(n_predators=10, n_preys=40,
                      alpha=1.1, beta=0.02, delta=0.02, gamma=0.008):
        """
        Discrete version of the Lotka–Volterra equations for predator-prey model. The equations are:

          preys(t + 1)       = alpha*preys(t)           - beta*preys*predators(t)
          predators(t + 1)   = delta*preys*predators(t) - gamma*predators(t)

        :param n_predators: Initial number of predators.
        :param n_preys: Initial number of preys.
        :param alpha: Prey reproduction factor.
        :param beta: Effective killing factor for predators.
        :param delta: Effective multiplication factor for predators based on prey consumption.
        :param gamma: Natural death rate for predators.

        :return: A predators and a preys model.
        """

        # preys     = alpha*preys           - beta*preys*predators
        # predators = delta*preys*predators - gamma*predators

        past_preys = PastEvent(1, undefined_value=n_preys, name='Past Preys', allow_learning=False)
        past_predators = PastEvent(1, undefined_value=n_predators, name='Past Predators', allow_learning=False)

        preys = ClipEvent((ConstantEvent(alpha) * past_preys - ConstantEvent(beta)*past_preys*past_predators),
                          min_value=0.0,
                          name="Preys")

        predators = ClipEvent((ConstantEvent(delta) * past_preys * past_predators - ConstantEvent(gamma) * past_predators),
                              min_value=0.0,
                              name="Predators")

        preys.parallel_to(predators)
        predators.parallel_to(preys)

        past_preys.refers_to(preys)
        past_predators.refers_to(predators)

        return predators, preys
