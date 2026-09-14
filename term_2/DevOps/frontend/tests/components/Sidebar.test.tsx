// frontend/tests/components/Sidebar.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Sidebar from '../../components/Sidebar';
import { User, Area } from '../../types';

const mockUser: User = { id: '1', full_name: 'Test User', email: 'test@example.com' };
const mockAreas: Area[] = [
  { id: '1', name: 'Work', color: 'bg-red-500' },
  { id: '2', name: 'Personal', color: 'bg-blue-500' },
];

describe('Sidebar Component', () => {
  const onNavigate = vi.fn();
  const onLogout = vi.fn();
  const onCreateArea = vi.fn(() => Promise.resolve());

  it('renders correctly with user data', () => {
    render(
      <Sidebar
        currentPage="dashboard"
        onNavigate={onNavigate}
        user={mockUser}
        onLogout={onLogout}
        areas={mockAreas}
        onCreateArea={onCreateArea}
        activeTasksCount={5}
      />
    );

    expect(screen.getByText('Test User')).toBeInTheDocument();
    expect(screen.getByText('Work')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('navigates to a new page when a nav item is clicked', async () => {
    render(
        <Sidebar
          currentPage="dashboard"
          onNavigate={onNavigate}
          user={mockUser}
          onLogout={onLogout}
          areas={mockAreas}
          onCreateArea={onCreateArea}
          activeTasksCount={5}
        />
      );

    await userEvent.click(screen.getByText('Tasks'));
    expect(onNavigate).toHaveBeenCalledWith('tasks');
  });

  it('navigates to an area when an area item is clicked', async () => {
    render(
        <Sidebar
          currentPage="dashboard"
          onNavigate={onNavigate}
          user={mockUser}
          onLogout={onLogout}
          areas={mockAreas}
          onCreateArea={onCreateArea}
          activeTasksCount={5}
        />
      );

    await userEvent.click(screen.getByText('Work'));
    expect(onNavigate).toHaveBeenCalledWith('area:1');
  });

  it('shows and handles logout', async () => {
    render(
        <Sidebar
          currentPage="dashboard"
          onNavigate={onNavigate}
          user={mockUser}
          onLogout={onLogout}
          areas={mockAreas}
          onCreateArea={onCreateArea}
          activeTasksCount={5}
        />
      );

    await userEvent.click(screen.getByText('Test User'));
    const logoutButton = screen.getByText('Log out');
    expect(logoutButton).toBeInTheDocument();
    await userEvent.click(logoutButton);
    expect(onLogout).toHaveBeenCalled();
  });

  it('shows create area input and creates an area on enter', async () => {
    onCreateArea.mockClear();
    render(
        <Sidebar
          currentPage="dashboard"
          onNavigate={onNavigate}
          user={mockUser}
          onLogout={onLogout}
          areas={mockAreas}
          onCreateArea={onCreateArea}
          activeTasksCount={5}
        />
      );

    await userEvent.click(screen.getByText('Create area'));
    const input = screen.getByPlaceholderText('Name');
    expect(input).toBeInTheDocument();
    await userEvent.type(input, 'New Area');
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
    expect(onCreateArea).toHaveBeenCalledWith('New Area', expect.any(String));
  });

  it('hides create area input on escape', async () => {
    render(
        <Sidebar
          currentPage="dashboard"
          onNavigate={onNavigate}
          user={mockUser}
          onLogout={onLogout}
          areas={mockAreas}
          onCreateArea={onCreateArea}
          activeTasksCount={5}
        />
      );

    await userEvent.click(screen.getByText('Create area'));
    const input = screen.getByPlaceholderText('Name');
    expect(input).toBeInTheDocument();
    fireEvent.keyDown(input, { key: 'Escape', code: 'Escape' });
    expect(screen.queryByPlaceholderText('Name')).not.toBeInTheDocument();
  });

  it('shows color palette and selects a color', async () => {
    render(
        <Sidebar
          currentPage="dashboard"
          onNavigate={onNavigate}
          user={mockUser}
          onLogout={onLogout}
          areas={mockAreas}
          onCreateArea={onCreateArea}
          activeTasksCount={5}
        />
      );

    await userEvent.click(screen.getByText('Create area'));
    const colorDot = screen.getByTestId('color-dot');
    await userEvent.click(colorDot);
    
    const colorPalette = screen.getByTestId('color-palette');
    expect(colorPalette).toBeInTheDocument();

    const colorOptions = colorPalette.querySelectorAll('div[class*="bg-"]');
    expect(colorOptions.length).toBe(9);
    await userEvent.click(colorOptions[3]);
    
    expect(screen.queryByTestId('color-palette')).not.toBeInTheDocument();
  });
});
