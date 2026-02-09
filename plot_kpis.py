import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

def generate_plots():
    """
    Reads all csv files from the calculated_kpis directory, generates interactive plots,
    and creates an index.html file to view them.
    """
    # Create a directory to store the plots
    if not os.path.exists('gui/plots'):
        os.makedirs('gui/plots')

    # Get all csv files from the calculated_kpis directory
    path = 'calculated_kpis'
    all_files = glob.glob(os.path.join(path, "*.csv"))

    # Create an index.html file to link to all the plots
    with open('gui/index.html', 'w') as f:
        f.write('<html><head><title>KPI Plots</title></head><body>')
        f.write('<h1>KPI Plots</h1>')
        f.write('<ul>')

        for file in all_files:
            # Read the csv file
            df = pd.read_csv(file)

            # Get the file name without the extension
            file_name = os.path.splitext(os.path.basename(file))[0]

            plot_file_name = f'plots/{file_name}.html'

            if file_name == 'summary_kpis':
                # For summary_kpis, create a table
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[df[col] for col in df.columns],
                               fill_color='lavender',
                               align='left'))
                ])
                fig.update_layout(title_text=file_name)
                fig.write_html(f'gui/{plot_file_name}')
            else:
                # For other KPIs, create a line plot
                if 'timestamp' in df.columns:
                    fig = px.line(df, x='timestamp', y=df.columns[1:], title=file_name)
                    fig.write_html(f'gui/{plot_file_name}')
                else:
                    print(f"Skipping {file_name}.csv as it does not contain a 'timestamp' column for line plotting.")
                    continue # Skip adding to index.html if not plotted

            # Add a link to the plot/table in the index.html file
            f.write(f'<li><a href="{plot_file_name}">{file_name}</a></li>')

        f.write('</ul></body></html>')

    print("Plots generated successfully in the 'gui/plots' directory.")
    print("Open 'gui/index.html' in your browser to view the plots.")

if __name__ == '__main__':
    generate_plots()