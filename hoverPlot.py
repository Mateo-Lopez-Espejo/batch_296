import bokeh.plotting as bp
import bokeh.models as bm
from bokeh.palettes import Category10 as palette
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


'''this function is destined to deal with a pandas dataframe containing the cell name, and the two columns of 
data to be plotted agains each other'''

def hoverPlot (DataFrame, ID, x, y, plotname = 'temp_intr_plot'):
    source = bp.ColumnDataSource(data=DataFrame)
    bp.output_file('{}.html'.format(plotname))
    hover = bm.HoverTool()
    hover.tooltips = [(ID, '@{{{}}}'.format(ID)),  # this {{{ }}} shenanigans are to scape ( {{ ) a  curly brace on either side
                      (x, '@{{{}}}'.format(x)),
                      (y, '@{{{}}}'.format(y))]

    fig = bp.figure(plot_width=800, plot_height=800, title=plotname, toolbar_location="right",
                    toolbar_sticky=False, tools=[hover, bm.PanTool(), bm.WheelZoomTool(), bm.ResetTool()])
    # scatter
    fig.circle(x, y, size=10, color='black', source=source)

    # trendline
    linfit = stats.linregress(DataFrame[x], DataFrame[y])
    legend = 'slope: {} \n rvalue: {} \n pvalue {}'.format(linfit.slope, linfit.rvalue, linfit.pvalue)

    def linfunct(a):
        return a * linfit[0] + linfit[1]

    min = DataFrame[x].min()
    max = DataFrame[x].max()

    fig.line([min,max],[linfunct(min),linfunct(max)], color='red', legend = legend)

    # Formating
    fig.legend.location = 'top_left'
    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y


    bp.show(fig)

def hoverPlotv2(DataFrame, level1, ID = 'cellid', level2 = None, plotname = 'temp_scatter'):
    # Asumes starting point of a long format DB,
    # level1 parameter should have 2 values, plotted as the 2 dimentions of the scatterplot
    # level2 parameter can have any number of level values, ploted as different colors. keep in mind to not overcrowd.

    lv1vals = DataFrame.index.get_level_values(level1).unique()
    x = lv1vals[0] ; y = lv1vals [1]

    bp.output_file('{}.html'.format(plotname))
    hover = bm.HoverTool()
    hover.tooltips = [(ID, '@{{{}}}'.format(ID)),
                      (x, '@{{{}}}'.format(x)),
                      (y, '@{{{}}}'.format(y))]

    fig = bp.figure(plot_width=800, plot_height=800, title=plotname, toolbar_location="right",
                    toolbar_sticky=False, tools=[hover, bm.PanTool(), bm.WheelZoomTool(), bm.ResetTool()])

    colors = palette[9]


    if level2 != None:
        lv2vals = DataFrame.index.get_level_values(level2).unique()

        for ll, color in zip(lv2vals, colors):
            query = '{} == "{}"'.format(level2, ll)
            workingDF = DataFrame.query(query)
            workingDF = workingDF.reset_index()
            workingDF = workingDF.pivot(index=ID, columns = level1, values = 'values')
            workingDF = workingDF.reset_index(ID)
            source = bp.ColumnDataSource(data=workingDF)

            #scatter
            fig.circle(x, y, size=10, color=color, source=source)

            # trendline
            linfit = stats.linregress(workingDF[x], workingDF[y])
            legend = '{}: slope: {} \n rvalue: {} \n pvalue {}'.format(ll, linfit.slope, linfit.rvalue, linfit.pvalue)

            def linfunct(a):
                return a * linfit[0] + linfit[1]

            min = workingDF[x].min()
            max = workingDF[x].max()

            fig.line([min, max], [linfunct(min), linfunct(max)], color=color, legend=legend)


        bp.show(fig)

    elif level2 == None:
        workingDF = DataFrame.reset_index()
        workingDF = workingDF.pivot(index=ID, columns=level1, values='values')
        workingDF = workingDF.reset_index(ID)
        source = bp.ColumnDataSource(data=workingDF)

        # scatter
        fig.circle(lv1vals[0], lv1vals[1], size=10, color='black', source=source)

        # trendline
        linfit = stats.linregress(workingDF[x], workingDF[y])
        legend = 'slope: {} \n rvalue: {} \n pvalue {}'.format(linfit.slope, linfit.rvalue, linfit.pvalue)

        def linfunct(a):
            return a * linfit[0] + linfit[1]

        min = workingDF[x].min()
        max = workingDF[x].max()

        fig.line([min, max], [linfunct(min), linfunct(max)], color='black', legend=legend)

    fig.legend.location = 'top_left'
    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y

def hoverPlotv3 (x_in, y_in, ID, var = None, xlabel = None, ylabel = None, tittle = None):
    #x_in, y_in, ID and var should have the same number of elements, they are lists of lists
    if len(set([len(x_in), len(y_in), len(ID)])) != 1 :
        print ('error, inputs should have same length')
        return

    # defines lineare regression
    lines = list()
    for x, y in zip(x_in, y_in):
        linfit = stats.linregress(x,y)
        minx = np.min(x)
        maxx = np.max(x)
        fmin = minx * linfit[0] + linfit[1]
        fmax = maxx * linfit[0] + linfit[1]
        lines.append([[minx, maxx], [fmin, fmax]])

    lines = np.asarray(lines)

    # appends all elementes in single string for each var for plotting with a single scatter call

    varlen = list()
    allx = list()
    for ii,vv in enumerate(x_in):
        allx = allx + vv
        varlen = varlen + [ii] * (len(vv))
    print(varlen)

    ally = list()
    for ii, vv in enumerate(y_in):
        ally = ally + vv

    allID = list()
    for ii, vv in enumerate(ID):
        allID = allID + vv


    def onpick3(event):
        ind = event.ind[0]
        print('onpick3 scatter:', allID[ind], allx[ind], ally[ind])

    fig, ax = plt.subplots()
    ax.scatter(allx, ally, s= None, c=varlen, picker=True)
    fig.canvas.mpl_connect('pick_event', onpick3)

    for ii in range(lines.shape[0]):
        ax.scatter(lines[ii,0,:], lines[ii,1,:], s = None, c = [1,1], marker= None, linewidths = 1)


    plt.show()

#test DF, not working currently
'''x = [[1,2,3], [1,2,3]]
y = [[1,2,3], [2,3,4]]
id = [['a', 'b', 'c'],['a', 'b', 'c']]
hoverPlotv3(x,y,id)'''
