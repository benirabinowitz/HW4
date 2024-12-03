"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from .datasets.road_dataset import load_data


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        if isinstance(model, CNNPlanner):
            image = batch['image'].to(device)
            pred_waypoints = model(image=image)
        else:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            pred_waypoints = model(track_left=track_left, track_right=track_right)
            
        target_waypoints = batch['waypoints'].to(device)
        waypoints_mask = batch['waypoints_mask'].to(device)
        
        optimizer.zero_grad()
        loss = F.mse_loss(pred_waypoints[waypoints_mask], target_waypoints[waypoints_mask])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(model, CNNPlanner):
                image = batch['image'].to(device)
                pred_waypoints = model(image=image)
            else:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                pred_waypoints = model(track_left=track_left, track_right=track_right)
                
            target_waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            loss = F.mse_loss(pred_waypoints[waypoints_mask], target_waypoints[waypoints_mask])
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader = load_data(
        args.train_path,
        transform_pipeline=args.transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    print(f"Train loader size: {len(train_loader)}")
    
    val_loader = load_data(
        args.val_path,
        transform_pipeline=args.transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    print(f"Val loader size: {len(val_loader)}")
    
    if args.model == 'mlp':
        model = MLPPlanner(
            n_track=10,
            n_waypoints=3,
        )
    elif args.model == 'transformer':
        model = TransformerPlanner()
    else:
        model = CNNPlanner()
    print(f"Created model: {args.model}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model)
            print(f'New best model saved! Val Loss: {val_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'transformer', 'cnn'])
    parser.add_argument('--train_path', type=str, default='drive_data/train')
    parser.add_argument('--val_path', type=str, default='drive_data/val')
    parser.add_argument('--transform', type=str, default='default')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    main(args)
