from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from matplotlib.pyplot import figure

class HWES(nn.Module):
    def __init__(self):
        def __init__(self, sigma, window, density):
            super().__init__()
            self.sigma = sigma
            self.window = window
            self.density = density
            self.weight = self.gaussian().to(DEVICE)

    def getDataset(csv_file):
        '''Read CSV file using pandas'''
        T = pd.read_csv(csv_file, index_col=False)
        return T

    def printFigure(fac,w,h):
        '''Print plots of temporal factor'''
        num = len(time_list)
        plt.figure()
        figure(figsize=(w,h), dpi=80)
        for i in range(0,num):
            plt.subplot(1,num,i+1)
            fac[str(time_list[i])].plot()
            plt.title(f'{time_list[i]}th')

    def buildModelHWES(temp,train_indx,seasonp,n,zoom):
        '''Implement HWES and show results in graphs'''
        train = temp[:train_indx]
        test = temp[train_indx:]
        
        model = HWES(train, seasonal_periods=seasonp, trend='add',seasonal='mul')
        fit = model.fit(optimized=True,use_brute=True)
        print(fit.summary())
        forecast = fit.forecast(steps=n)

        ticks = range(len(train)+len(test))
        fig = plt.figure()
        past, = plt.plot(ticks[-zoom:-n], train[-zoom:-n], 'b.-', label='Traffic histroy')
        future, = plt.plot(ticks[-n:], test[-n:], 'r.-', label='Traffic future')
        predicted, = plt.plot(ticks[-n:], forecast, 'g.-', label='Traffic predicted')
        plt.legend()
        fig.show()
        plt.show()

        return forecast
