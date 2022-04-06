from Model import Model
import statsmodels.tsa.stattools as ts

class ModulatedMultidimensionalAtmosphericScrubbers(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2015, 8, 15)  # Set Start Date
        self.SetEndDate(2020, 8, 15)
        self.SetCash(100000)  # Set Strategy Cash
        self.SetBenchmark('SPY')
        
        self.ATickers = ['GLD']
        self.BTickers = ['SLV']
        self.ASymbols = []
        self.BSymbols = []
        for i in self.ATickers:
            self.ASymbols.append(self.AddEquity(i, Resolution.Daily).Symbol)
        for i in self.BTickers:
            self.BSymbols.append(self.AddEquity(i, Resolution.Daily).Symbol)
        self.SetWarmup(200)
        self.model = Model()
        
        # retrain our model periodically
        self.Train(self.DateRules.MonthStart('GLD'), self.TimeRules.Midnight, self.TrainModel)
        self.months = 0
        
    def OnData(self, data):
        for i in range(len(self.ASymbols)):
            self.model.Update(self.Time, data[self.ASymbols[i]].Close, data[self.BSymbols[i]].Close)
        
        if self.IsWarmingUp:
            return
        
        if not self.model.IsReady:
            return
        
        # if we aren't holding the portfolio and our model tells us to buy
        #   the portfolio, we buy the portfolio
        for i in range(len(self.ASymbols)):
            if not self.Portfolio.Invested and self.model.IsEnter() and self.model.coint:
                self.SetHoldings(self.ASymbols[i], 1) 
                self.SetHoldings(self.BSymbols[i], -self.model.AllocationB())
        # if we are holding the portfolio and our model tells us to sell
        #   the portfolio, we liquidate our holdings
            elif self.Portfolio.Invested and self.model.IsExit():
                self.Liquidate()
        
    def TrainModel(self):
        if not self.model.Ready2Train:
            return
        
        self.months += 1
        
        ts_A = np.array(list(self.model.close_A))[::-1]
        ts_B = np.array(list(self.model.close_B))[::-1]
        
        coin_result = ts.coint(ts_A, ts_B)
        if coin_result[1] <= 0.1:
            self.model.coint = True
            self.Debug(str(coin_result[1]) + " " + str(self.model.coint))
        else:
            self.model.coint = False
            self.Debug(str(coin_result[1]) + " " + str(self.model.coint))
        
        # only retrain every 7 months
        if self.months % 7 != 1:
            return
        
        self.model.Train()
        
    
        if not self.model.IsReady:
            self.Liquidate()
            return
            
        theta, mu, sigma = self.model.get_coefficients()
        
        self.Log('θ: ' + str(round(theta, 2)) + ' μ: ' + str(round(mu, 2)) + ' σ: ' + str(round(sigma, 2)))
