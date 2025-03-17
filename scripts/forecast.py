#!/usr/bin/env python3
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import os


def predict_and_plot_cluster_aggregates(aggregated_data_path, output_plot_path, metric):
    data = pd.read_parquet(aggregated_data_path)
    data.index = pd.to_datetime(data.index)

    normalise_date = pd.Timestamp("2018-01-01")
    split_date = pd.Timestamp("2020-01-01")

    plt.figure(figsize=(12, 6))
    for cluster in data.columns:
        print(f"Processing {cluster}...")

        cluster_data = data[[cluster]].reset_index()
        cluster_data.columns = ["ds", "y"]

        # train_data = cluster_data[cluster_data["ds"] < split_date]
        train_data = cluster_data[(normalise_date <= cluster_data["ds"]) & (cluster_data["ds"] < split_date)]

        test_data = cluster_data[cluster_data["ds"] >= split_date]

        model = Prophet(weekly_seasonality=True)
        
        # model = Prophet(changepoint_prior_scale=0.5, weekly_seasonality=True)

        # train_data["cap"] = 1.0  # or another appropriate capacity value
        # model = Prophet(growth='logistic', weekly_seasonality=True)

        model.fit(train_data)

        # future = model.make_future_dataframe(periods=len(test_data), freq="W")
        future = model.make_future_dataframe(periods=len(test_data), freq='W-SUN')

        forecast = model.predict(future)

        predicted = forecast[forecast["ds"].isin(test_data["ds"])]["yhat"]
        mae = mean_absolute_error(test_data["y"], predicted)
        rmse = sqrt(mean_squared_error(test_data["y"], predicted))
        print(f"Forecast MAE for {cluster}: {mae:.2f}")
        print(f"Forecast RMSE for {cluster}: {rmse:.2f}")

        plt.plot(cluster_data["ds"], cluster_data["y"], label=f"Observed {cluster}", alpha=0.6)
        
        forecast_from_2020 = forecast[forecast["ds"] >= split_date]
        plt.plot(forecast_from_2020["ds"], forecast_from_2020["yhat"], linestyle="--", 
                 label=f"Predicted {cluster}", alpha=0.8)

        plt.axvline(split_date, color="red", linestyle="--", 
                    label="2020-01-01" if cluster == data.columns[0] else "")

    plt.title(f"Time Series Prediction for {metric} from 2020 Onwards")
    plt.xlabel("Date")
    plt.ylabel(f"Aggregate {metric}")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()

    print(f"Prediction plot saved to {output_plot_path}")


def predict_and_plot_time_series(data_path, metric_name, output_plot_path):
    data = pd.read_parquet(data_path)

    data.columns = pd.to_datetime(data.columns, format="%d/%m/%Y")

    data = data.T 

    if metric_name not in data.columns:
        raise ValueError(
            f"Metric '{metric_name}' not found. "
            f"Available metrics: {list(data.columns)}"
        )

    data = data[[metric_name]]

    data.rename(columns={metric_name: 'y'}, inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'ds'}, inplace=True)

    split_date = pd.Timestamp("2020-01-01")
    train_data = data[data['ds'] < split_date].copy()
    test_data  = data[data['ds'] >= split_date].copy()

    model = Prophet(weekly_seasonality=True)
    model.fit(train_data)

    future = model.make_future_dataframe(periods=len(test_data), freq="W")
    forecast = model.predict(future)

    forecast_2020plus = forecast[forecast['ds'] >= split_date]

    y_true = test_data['y'].values
    y_pred = forecast_2020plus['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    print(f"Forecast MAE for {metric_name}: {mae:.2f}")
    print(f"Forecast RMSE for {metric_name}: {rmse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(data['ds'], data['y'], label=f"Observed {metric_name}", alpha=0.6)
    plt.plot(forecast_2020plus['ds'], forecast_2020plus['yhat'],
             linestyle="--", label=f"Predicted {metric_name}", alpha=0.8)

    plt.axvline(split_date, color="red", linestyle="--", label="2020-01-01")

    plt.title(f"Time Series Prediction for {metric_name} from 2020 Onwards")
    plt.xlabel("Date")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_plot_path) or ".", exist_ok=True)
    plt.savefig(output_plot_path)
    plt.show()
    print(f"Prediction plot saved to {output_plot_path}")


def train_and_validate_prophet(train_df, test_df, freq='D'):
    model = Prophet(weekly_seasonality=True)
    model.fit(train_df)

    last_train_date = train_df['ds'].max()
    last_test_date = test_df['ds'].max()

    horizon_days = (last_test_date - last_train_date).days

    future = model.make_future_dataframe(periods=horizon_days, freq=freq)
    forecast = model.predict(future)

    test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
    mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])
    rmse = sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))

    return model, forecast, mae, rmse


def predict_and_plot_single_df(
    aggregated_data_path,
    output_plot_path,
    train_start='2019-07-01',
    train_end='2019-09-30',
    test_start='2019-10-01',
    test_end='2019-12-31',
    post_test_start='2020-01-01',
    freq='D'
):
    data = pd.read_parquet(aggregated_data_path)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts   = pd.Timestamp(train_end)
    test_start_ts  = pd.Timestamp(test_start)
    test_end_ts    = pd.Timestamp(test_end)
    post_test_ts   = pd.Timestamp(post_test_start)

    clusters = data.columns
    n_clusters = len(clusters)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 5*n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes] 

    metrics = {
        'cluster': [],
        'MAE': [],
        'RMSE': []
    }

    for i, cluster in enumerate(clusters):
        ax = axes[i]

        cluster_df = data[[cluster]].reset_index()
        cluster_df.columns = ['ds', 'y']

        train_mask = (cluster_df['ds'] >= train_start_ts) & (cluster_df['ds'] <= train_end_ts)
        test_mask =  (cluster_df['ds'] >= test_start_ts)  & (cluster_df['ds'] <= test_end_ts)

        train_data = cluster_df.loc[train_mask]
        test_data  = cluster_df.loc[test_mask]

        try:
            model, in_sample_forecast, mae, rmse = train_and_validate_prophet(train_data, test_data, freq=freq)
            metrics['cluster'].append(cluster)
            metrics['MAE'].append(mae)
            metrics['RMSE'].append(rmse)
        except ValueError as e:
            print(f"Skipping cluster '{cluster}' due to error: {e}")
            metrics['cluster'].append(cluster)
            metrics['MAE'].append(None)
            metrics['RMSE'].append(None)
            continue

        print(f"\nCluster '{cluster}': MAE={mae:.2f}, RMSE={rmse:.2f}")

        last_test_date = test_end_ts
        max_date = cluster_df['ds'].max()
        days_to_forecast = (max_date - last_test_date).days

        if days_to_forecast > 0:
            future_df = model.make_future_dataframe(periods=days_to_forecast, freq=freq)
            final_forecast = model.predict(future_df)
        else:
            final_forecast = in_sample_forecast

        ax.plot(cluster_df['ds'], cluster_df['y'], label='Observed', color='black', alpha=0.6)

        ax.plot(in_sample_forecast['ds'], in_sample_forecast['yhat'], 
                color='blue', linestyle='--', label='Train+Test Prediction')

        ax.plot(final_forecast['ds'], final_forecast['yhat'], 
                color='red', linestyle='--', label='Post-Test Prediction')
        
        ax.axvline(train_end_ts, color='orange', linestyle='--', label='Train End')
        ax.axvline(test_end_ts,  color='green',  linestyle='--', label='Test End')

        ax.set_xlim([train_start_ts - pd.Timedelta(days=5), max_date + pd.Timedelta(days=5)])

        ax.set_title(f"Cluster: {cluster}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Metric")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {output_plot_path}")

    metrics_df = pd.DataFrame(metrics)
    print("Validation Metrics:")
    print(metrics_df)
    return metrics_df


if __name__ == "__main__":

    metrics_dict = {
        "Percentage of Posts with Comments": "comments_percentage",
        "Raw Number of Posts": "submissions",
        "Raw Number of Comments": "comments",
        "Post Lifespan (Mean in hours)": "lifespan",
        "Average Response Time with Cutoff (Minutes)": "response",
        "Average Sentiment": "average_sentiment",
        "Positive Sentiment Count": "positive_sentiment",
        "Negative Sentiment Count": "negative_sentiment",
    }

    # predict_and_plot_cluster_aggregates(
    #     aggregated_data_path="../cluster_aggregated_metrics/aggregated_comment_percentage.parquet",
    #     output_plot_path="../clustered_prophet_plots/comment_percentage_cluster_forecast_2.png"
    # )

    # for metric_key, metric_name in metrics_dict.items():
    #     predict_and_plot_cluster_aggregates(
    #         aggregated_data_path=f"../cluster_fns_aggregated_metrics/aggregated_{metric_name}.parquet",
    #         output_plot_path=f"../clustered_fns_prophet_plots/{metric_name}_cluster_forecast.png",
    #         metric=metric_key
    #     )
    
    # for metric_key, metric_name in metrics_dict.items():
    #     predict_and_plot_cluster_aggregates(
    #         aggregated_data_path=f"../cluster_fns_aggregated_fns_metrics/aggregated_{metric_name}.parquet",
    #         output_plot_path=f"../clustered_fns_prophet_plots_fns_metrics/{metric_name}_cluster_forecast.png",
    #         metric=metric_key
    #     )

    for metric_key, metric_name in metrics_dict.items():
        predict_and_plot_cluster_aggregates(
            aggregated_data_path=f"../cluster_fns_aggregated_fns_metrics2/aggregated_{metric_name}.parquet",
            output_plot_path=f"../clustered_fns_prophet_plots_fns_metrics2/{metric_name}_cluster_forecast.png",
            metric=metric_key
        )

    # for metric_key, metric_name in metrics_dict.items():
    #     predict_and_plot_cluster_aggregates(
    #         aggregated_data_path=f"../cluster_aggregated_metrics/aggregated_{metric_name}.parquet",
    #         output_plot_path=f"../clustered_prophet_plots/{metric_name}_cluster_forecast.png",
    #         metric=metric_key
    #     )

    # for metric_key, metric_name in metrics_dict.items():
    #     predict_and_plot_cluster_aggregates(
    #         aggregated_data_path=f"../cluster_aggregated_metrics/aggregated_{metric_name}.parquet",
    #         output_plot_path=f"../clustered_prophet_plots/{metric_name}_cluster_forecast.png",
    #         metric=metric_key
    #     )